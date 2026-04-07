"""DOGE/SOL pairs (z=3.0) — taker-profitable, validated."""
PAIR_A = "DOGE/USDT"
PAIR_B = "SOL/USDT"
LOOKBACK = 80
ENTRY_Z = 3.0
EXIT_Z = 0.5

def get_params():
    return {"pair_a": PAIR_A, "pair_b": PAIR_B, "lookback": LOOKBACK, "entry_z": ENTRY_Z, "exit_z": EXIT_Z}
