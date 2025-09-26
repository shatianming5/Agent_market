#!/usr/bin/env python
"""Lightweight Uniswap v2/v3 indexer for recent swaps.

Designed for workshop/demo use:
- Reads pool list from conf/dex.yaml (no need to scan entire chain)
- Handles missing RPC credentials gracefully
- Stores swaps as parquet under data/raw/dex/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from web3 import Web3
from web3._utils.events import get_event_data
import yaml

load_dotenv()

# Event topics
V2_SWAP_TOPIC = Web3.keccak(text='Swap(address,uint256,uint256,uint256,uint256,address)')
V3_SWAP_TOPIC = Web3.keccak(text='Swap(address,address,int256,int256,uint160,uint128,int24)')

# Minimal ABIs for decoding
V2_SWAP_ABI = json.loads('[{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"sender","type":"address"},{"indexed":false,"internalType":"uint256","name":"amount0In","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1In","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount0Out","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"amount1Out","type":"uint256"},{"indexed":true,"internalType":"address","name":"to","type":"address"}],"name":"Swap","type":"event"}]')
V3_SWAP_ABI = json.loads('[{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"sender","type":"address"},{"indexed":true,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"int256","name":"amount0","type":"int256"},{"indexed":false,"internalType":"int256","name":"amount1","type":"int256"},{"indexed":false,"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},{"indexed":false,"internalType":"uint128","name":"liquidity","type":"uint128"},{"indexed":false,"internalType":"int24","name":"tick","type":"int24"}],"name":"Swap","type":"event"}]')


@dataclass
class PoolConfig:
    address: str
    token0: str
    token1: str
    token0_decimals: int
    token1_decimals: int


@dataclass
class DexConfig:
    name: str
    version: str
    block_window: int
    pools: List[PoolConfig]


@dataclass
class Config:
    chain_id: int
    rpc_url_env: str
    confirmations: int
    dexes: List[DexConfig]


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    dexes = []
    for dex in raw.get('dexes', []):
        pools = [PoolConfig(**pool) for pool in dex.get('pools', [])]
        dexes.append(
            DexConfig(
                name=dex['name'],
                version=dex['version'],
                block_window=dex.get('block_window', 2000),
                pools=pools,
            )
        )
    return Config(
        chain_id=raw['chain_id'],
        rpc_url_env=raw.get('rpc_url_env', 'RPC_URL_ETHEREUM'),
        confirmations=raw.get('confirmations', 6),
        dexes=dexes,
    )



block_timestamp_cache: Dict[int, int] = {}


def get_block_timestamp(w3: Web3, block_number: int) -> int:
    if block_number in block_timestamp_cache:
        return block_timestamp_cache[block_number]
    block = w3.eth.get_block(block_number)
    ts = block['timestamp']
    block_timestamp_cache[block_number] = ts
    return ts

@retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(4))
def get_logs(w3: Web3, params: Dict) -> List[Dict]:
    return w3.eth.get_logs(params)


def decode_swap(w3: Web3, dex_version: str, log: Dict) -> Dict:
    abi = V3_SWAP_ABI if dex_version == 'v3' else V2_SWAP_ABI
    return get_event_data(w3.codec, abi[0], log)


def fetch_swaps_for_pool(
    w3: Web3,
    pool: PoolConfig,
    version: str,
    latest_block: int,
    confirmations: int,
    block_window: int,
) -> List[Dict]:
    to_block = latest_block - confirmations
    from_block = max(0, to_block - block_window)
    topic = V3_SWAP_TOPIC if version == 'v3' else V2_SWAP_TOPIC
    step = 500
    records: List[Dict] = []
    for start in range(from_block, to_block + 1, step):
        end = min(start + step - 1, to_block)
        params = {
            'fromBlock': start,
            'toBlock': end,
            'address': Web3.to_checksum_address(pool.address),
            'topics': [topic],
        }
        try:
            logs = get_logs(w3, params)
        except Exception as exc:
            if '429' in str(exc):
                print(f'Received 429 for blocks {start}-{end}, sleeping...')
                time.sleep(2)
                continue
            print(f"Failed fetching logs {start}-{end} for {pool.address}: {exc}")
            continue
        for log in logs:
            data = decode_swap(w3, version, log)
            args = data['args']
            if version == 'v3':
                amount0 = int(args['amount0'])
                amount1 = int(args['amount1'])
                sqrt_price = int(args['sqrtPriceX96'])
                price = (sqrt_price / (1 << 96)) ** 2
            else:
                amount0 = int(args['amount0Out']) - int(args['amount0In'])
                amount1 = int(args['amount1Out']) - int(args['amount1In'])
                # Use human-readable ratio when possible
                human0 = amount0 / (10 ** pool.token0_decimals) if pool.token0_decimals else 0
                human1 = amount1 / (10 ** pool.token1_decimals) if pool.token1_decimals else 0
                price = abs(human1 / human0) if human0 else None
            records.append(
                {
                    'pool': pool.address.lower(),
                    'token0': pool.token0.lower(),
                    'token1': pool.token1.lower(),
                    'token0_decimals': pool.token0_decimals,
                    'token1_decimals': pool.token1_decimals,
                    'tx_hash': log['transactionHash'].hex(),
                    'log_index': log['logIndex'],
                    'block_number': log['blockNumber'],
                    'block_time': get_block_timestamp(w3, log['blockNumber']),
                    'amount0': amount0,
                    'amount1': amount1,
                    'price': price,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description='Index recent DEX swaps')
    parser.add_argument('--config', type=Path, default=Path('conf/dex.yaml'))
    parser.add_argument('--output', type=Path, default=Path('data/raw/dex'))
    args = parser.parse_args()

    if not args.config.exists():
        print(f'Config file {args.config} not found; aborting.')
        return
    cfg = load_config(args.config)

    rpc_url = os.getenv(cfg.rpc_url_env)
    if not rpc_url:
        print(f'Environment variable {cfg.rpc_url_env} not set; skipping indexing.')
        print("Please add it to .env and re-run.")
        return

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f'Failed to connect to RPC endpoint {rpc_url}; aborting.')
        return

    latest_block = w3.eth.block_number
    print(f'Connected to chain {cfg.chain_id}, latest block {latest_block}')

    args.output.mkdir(parents=True, exist_ok=True)

    for dex in cfg.dexes:
        all_records: List[Dict] = []
        print(f"Processing {dex.name} ({dex.version}) with {len(dex.pools)} pools")
        for pool in tqdm(dex.pools, desc=dex.name):
            records = fetch_swaps_for_pool(
                w3,
                pool,
                dex.version,
                latest_block,
                cfg.confirmations,
                dex.block_window,
            )
            all_records.extend(records)
        if not all_records:
            print(f"No swaps collected for {dex.name} in the last {dex.block_window} blocks.")
            continue
        df = pd.DataFrame(all_records)
        df['block_time'] = pd.to_datetime(df['block_time'], unit='s', utc=True)
        df['amount0_norm'] = df['amount0'] / (10 ** df['token0_decimals'])
        df['amount1_norm'] = df['amount1'] / (10 ** df['token1_decimals'])
        df['amount0'] = df['amount0'].astype('object').astype(str)
        df['amount1'] = df['amount1'].astype('object').astype(str)
        out_path = args.output / f"{dex.name}_swaps.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} swaps -> {out_path}")


if __name__ == '__main__':
    main()
