"""Template: Cross-Sectional Momentum."""
ASSETS = ["BTC","ETH","SOL","DOGE","XRP","AVAX","ADA","DOT","LINK"]
LOOKBACK_HOURS = 168; TOP_N = 3; REBALANCE_EVERY = 168
def get_params():
    return {"assets": ASSETS, "lookback": LOOKBACK_HOURS, "top_n": TOP_N}
