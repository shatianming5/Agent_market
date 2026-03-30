"""Template: Funding Rate Signal.
When perpetual funding rate is extremely negative, go long (shorts paying longs).
NOTE: Requires futures funding rate data feed.
This is a SIGNAL FILE for custom backtest, not a freqtrade strategy."""
PAIR = "BTC/USDT"
FUNDING_THRESHOLD_LONG = -0.01   # very negative funding = longs get paid
FUNDING_THRESHOLD_SHORT = 0.03   # very positive funding = shorts get paid
LOOKBACK_HOURS = 24
TIMEFRAME = "1h"

def get_params():
    return {
        "pair": PAIR,
        "long_threshold": FUNDING_THRESHOLD_LONG,
        "short_threshold": FUNDING_THRESHOLD_SHORT,
        "lookback": LOOKBACK_HOURS,
        "timeframe": TIMEFRAME,
        "data_required": "funding_rate (not available in spot, needs futures API)",
    }
