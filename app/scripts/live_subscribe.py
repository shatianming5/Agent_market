#!/usr/bin/env python
"""Poll swap logs incrementally to emulate live feed when WebSocket access is unavailable."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import pandas as pd
from web3 import Web3

load_dotenv()

V2_SWAP_TOPIC = Web3.keccak(text='Swap(address,uint256,uint256,uint256,uint256,address)').hex()
V3_SWAP_TOPIC = Web3.keccak(text='Swap(address,address,int256,int256,uint160,uint128,int24)').hex()


def load_addresses(source: Path) -> List[str]:
    if not source.exists():
        return []
    if source.suffix == '.parquet':
        df = pd.read_parquet(source)
        for col in ['pool', 'pair', 'address']:
            if col in df.columns:
                return [Web3.to_checksum_address(addr) for addr in df[col].dropna().unique()]
        return []
    if source.suffix in {'.json', '.txt'}:
        data = json.loads(source.read_text(encoding='utf-8'))
        return [Web3.to_checksum_address(addr) for addr in data]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description='HTTP polling fallback for swap logs')
    parser.add_argument('--config', type=Path, default=Path('conf/dex.yaml'))
    parser.add_argument('--addresses', type=Path, default=Path('data/raw/dex/uniswap-v3_pools.parquet'))
    parser.add_argument('--max-events', type=int, default=20)
    parser.add_argument('--poll-interval', type=float, default=5.0)
    parser.add_argument('--output', type=Path, default=Path('data/raw/dex/live_swaps_poll.jsonl'))
    args = parser.parse_args()

    if not args.config.exists():
        print(f'Config file {args.config} not found; aborting.')
        return
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    rpc_env = cfg.get('rpc_url_env', 'RPC_URL_ETHEREUM')
    rpc_url = os.getenv(rpc_env)
    if not rpc_url:
        print(f'Environment variable {rpc_env} not set; skipping live polling.')
        return

    addresses = load_addresses(args.addresses)
    if not addresses:
        print(f'No pool addresses found in {args.addresses}; run dex_indexer first or supply a JSON list.')
        return

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f'Failed to connect to RPC endpoint: {rpc_url}')
        return

    confirmations = cfg.get('confirmations', 12)
    current_block = w3.eth.block_number
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topics = [[V2_SWAP_TOPIC, V3_SWAP_TOPIC]]

    stop_flag = False

    def handle_sigint(signum, frame):
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, handle_sigint)

    collected = 0
    with output_path.open('a', encoding='utf-8') as fh:
        while not stop_flag and collected < args.max_events:
            latest = w3.eth.block_number - confirmations
            if latest <= current_block:
                time.sleep(args.poll_interval)
                continue
            params = {
                'fromBlock': current_block + 1,
                'toBlock': latest,
                'address': addresses,
                'topics': topics,
            }
            try:
                logs = w3.eth.get_logs(params)
            except Exception as exc:
                print(f'get_logs error: {exc}; backing off...')
                time.sleep(args.poll_interval * 2)
                continue
            for log in logs:
                record = {
                    'blockNumber': log['blockNumber'],
                    'transactionHash': log['transactionHash'].hex(),
                    'logIndex': log['logIndex'],
                    'address': log['address'],
                    'topics': [topic.hex() for topic in log['topics']],
                }
                record = {
                    'blockNumber': log['blockNumber'],
                    'transactionHash': log['transactionHash'].hex(),
                    'logIndex': log['logIndex'],
                    'address': log['address'],
                    'topics': [topic.hex() for topic in log['topics']],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(args.poll_interval)
    print(f'Polling finished, collected {collected} events -> {output_path}')


if __name__ == '__main__':
    import yaml
    main()
