# é¡¹ç›®è¯´æ˜Ž

æœ¬ä»“åº“åŒ…å«ä¸¤ä¸ªå¯ç›´æŽ¥è¿è¡Œçš„æ•°æ®æµç¨‹ï¼š

1. **åŠ å¯†è´§å¸è¡Œæƒ…å›žæµ‹ç®¡çº¿**ï¼šä½¿ç”¨ CCXT æ‹‰å– BinanceUS è¡Œæƒ…ã€é…åˆè‡ªç ”æ¸…æ´—è„šæœ¬ä¸Ž vectorbt å›žæµ‹ã€‚
2. **æ–°é—» + Xï¼ˆåŽŸ Twitterï¼‰æ•°æ®é‡‡é›†ç®¡çº¿**ï¼šåˆè§„æŽ¥å…¥ RSS/News Sitemap ä¸Ž X APIï¼Œç»Ÿä¸€æ¸…æ´—å¹¶è¾“å‡ºåˆ†æžæŠ¥è¡¨ã€‚

---

## çŽ¯å¢ƒå‡†å¤‡

```powershell
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -U -r requirements.txt
```

> æç¤ºï¼šé¦–æ¬¡å®‰è£…è€—æ—¶è¾ƒé•¿ï¼ŒåŽç»­é‡å¤æ‰§è¡Œä¼šå¿«é€Ÿå®Œæˆã€‚

---

## åŠ å¯†è´§å¸è¡Œæƒ…ä¸Žå›žæµ‹æµç¨‹

| æ­¥éª¤ | å‘½ä»¤ | è¯´æ˜Ž |
| --- | --- | --- |
| 1 | `python scripts/fetch_ccxt_ohlcv.py --conf conf/symbols.yaml` | é€šè¿‡ CCXT å¢žé‡æ‹‰å– BinanceUS çŽ°è´§ 1h K çº¿ï¼Œè‡ªåŠ¨å‰”é™¤æœªæ”¶ç›˜èœ¡çƒ›ï¼Œè¾“å‡ºåˆ° `data/raw/exchange=...` |
| 2 | `python scripts/fetch_binance_bulk.py --conf conf/symbols.yaml --limit-months 4` | å¯é€‰ï¼šä¸‹è½½ Binance Data Vision æœˆåº¦ ZIPï¼Œæ ¡éªŒ CHECKSUM åŽè¡¥é½åŽ†å² |
| 3 | `python scripts/clean_ohlcv.py --conf conf/symbols.yaml` | åˆå¹¶ CCXT ä¸ŽåŽ†å²åŒ…ï¼ŒåŽ»é‡ã€æŽ’åºã€æŒ‰æœˆåˆ†åŒºå†™å…¥ `data/clean/exchange=...` |
| 4 | `python scripts/dq_report.py --mode ohlcv --conf conf/symbols.yaml` | ç”Ÿæˆè¡Œæƒ…æ•°æ®è¦†ç›–çŽ‡æŠ¥å‘Šï¼ˆç¼ºå£ã€æœ€å¤§é—´éš”ã€èµ·æ­¢æ—¶é—´ç­‰ï¼‰ |
| 5 | `python backtests/vbt_ma_rsi.py --conf conf/symbols.yaml` | ä½¿ç”¨ vectorbt çš„ MA+RSI ç­–ç•¥ç½‘æ ¼å›žæµ‹ï¼Œç»“æžœä¿å­˜åˆ° `backtests/results/*.parquet` |

**é»˜è®¤é…ç½® `conf/symbols.yaml`**

```yaml
exchange: binanceus
type: spot
symbols:
  - BTC/USDT
  - ETH/USDT
timeframes:
  - 1h
start: "2024-01-01"
end: null
store_as: parquet
```

å¦‚éœ€å¢žåŠ äº¤æ˜“å¯¹æˆ–æ—¶é—´ç²’åº¦ï¼Œç›´æŽ¥ä¿®æ”¹ä¸Šè¿°æ–‡ä»¶å¹¶é‡æ–°æ‰§è¡Œæµç¨‹ã€‚

---

## æ–°é—» + X æ•°æ®é‡‡é›†æµç¨‹

| æ­¥éª¤ | å‘½ä»¤ | è¯´æ˜Ž |
| --- | --- | --- |
| 1 | `python scripts/news_harvester.py --config conf/feeds.yaml` | å¼‚æ­¥æŠ“å– RSS ä¸Ž News Sitemapï¼Œéµå¾ª robots.txtï¼Œä½¿ç”¨ trafilatura æŠ½å–æ­£æ–‡ï¼Œè¾“å‡º `data/raw/news/news_*.parquet` |
| 2 | `python scripts/x_recent_search.py --max-results 20` | è°ƒç”¨ X Recent Search APIï¼ˆéœ€åœ¨ `.env` å¡«å†™ `X_BEARER_TOKEN`ï¼‰ï¼Œæœªé…ç½®æ—¶ä¼šæç¤ºå¹¶è·³è¿‡ |
| 3 | `python scripts/x_stream.py --max-messages 20` | è°ƒç”¨ X Filtered Streamï¼ˆéœ€ `X_STREAM_TOKEN` æˆ– `X_BEARER_TOKEN`ï¼‰ï¼Œå¯æ ¹æ® `conf/x_rules.yaml` è‡ªåŠ¨åŒæ­¥è§„åˆ™ |
| 4 | `python scripts/normalize.py` | åˆå¹¶æ–°é—»ä¸Ž X æ•°æ®ï¼Œåˆ©ç”¨ SimHash è¿‘é‡è¿‡æ»¤ï¼Œè¾“å‡ºç»Ÿä¸€ç»“æž„ `data/clean/news/all.parquet` |
| 5 | `python scripts/dq_report.py --mode news` | ç»Ÿè®¡æ¥æºã€è¯­è¨€ã€å‘å¸ƒæ—¶é—´åŒºé—´ã€é‡å¤ URL ç­‰è´¨é‡æŒ‡æ ‡ |

**é…ç½®æ–‡ä»¶ç¤ºä¾‹**

- `conf/feeds.yaml`ï¼šå®šä¹‰ RSSã€News Sitemapã€æ˜¯å¦å¯ç”¨ NewsAPI/GDELTã€‚
- `conf/x_rules.yaml`ï¼šX è¿‡æ»¤æµè§„åˆ™ï¼Œæ”¯æŒå¤šä¸»é¢˜ã€‚
- `.env`ï¼šå­˜æ”¾ X/NewsAPI/GDELT ç­‰å¯†é’¥ï¼ˆç¤ºä¾‹å·²æä¾›å ä½å­—æ®µï¼‰ã€‚

> X å¹³å°ä¸¥æ ¼é™åˆ¶æ•°æ®ç”¨é€”ä¸Žè¿”å›žé‡ï¼Œå¿…é¡»éµå®ˆå®˜æ–¹å¼€å‘è€…åè®®ï¼Œæ”¶åˆ°åˆ é™¤äº‹ä»¶éœ€åŠæ—¶åŒæ­¥åˆ é™¤æœ¬åœ°æ•°æ®ã€‚

---

## æ•°æ®è¾“å‡ºç»“æž„

- `data/raw/news/`ï¼šæ–°é—»åŽŸå§‹ parquetï¼ˆå«æ­£æ–‡ã€æ‘˜è¦ã€å…ƒæ•°æ®ã€åŽŸå§‹ JSON æ–‡æœ¬ï¼‰ã€‚
- `data/raw/x/`ï¼šRecent Search `jsonl` ä¸Ž Filtered Stream `ndjson`ã€‚
- `data/clean/news/all.parquet`ï¼šç»Ÿä¸€åŽçš„æ–°é—»/X æ•°æ®é›†ï¼Œå­—æ®µåŒ…æ‹¬ `source`ã€`url`ã€`title`ã€`published_at`ã€`text`ã€`tags`ã€`raw_json` ç­‰ã€‚
- `data/clean/exchange=.../`ï¼šæŒ‰æœˆåˆ†åŒºçš„è¡Œæƒ…æ•°æ®ã€‚
- `backtests/results/*.parquet`ï¼švectorbt å›žæµ‹æŒ‡æ ‡è¡¨ã€‚

---

## å¸¸è§æ‰©å±•

1. **è°ƒåº¦**ï¼šç»“åˆ `cron`ã€`APScheduler` æˆ– Prefect å®žçŽ°å®šæ—¶é‡‡é›†ä¸Žå›žæµ‹ã€‚
2. **åˆ†æž**ï¼šä½¿ç”¨ DuckDB/Polars ç›´æŽ¥è¯»å– parquetï¼Œæˆ–è½åœ°è‡³ PostgreSQL + OpenSearch å¹¶åœ¨ Superset/Metabase æ­å»ºä»ªè¡¨ç›˜ã€‚
3. **NLP å¯ŒåŒ–**ï¼šåœ¨ `scripts/normalize.py` ä¸­æ‰©å±• spaCy/HF æ¨¡åž‹å®žçŽ°å®žä½“æŠ½å–ã€æƒ…æ„Ÿåˆ†æžã€æ‘˜è¦ç­‰ã€‚
4. **é£Žé™©æŽ§åˆ¶**ï¼šåœ¨å›žæµ‹è„šæœ¬ä¸­åŠ å…¥ä»“ä½ç®¡ç†ã€æ»‘ç‚¹æ•æ„Ÿåº¦æµ‹è¯•ã€åˆ†å±‚èµ°æ ·éªŒè¯ã€‚

---

## åˆè§„æé†’

- **X API**ï¼šä»…ä½¿ç”¨å®˜æ–¹ç«¯ç‚¹ï¼Œéµå®ˆé€ŸçŽ‡ä¸Žå†…å®¹ä½¿ç”¨æ”¿ç­–ï¼Œç¦æ­¢æŠ“å–ç½‘é¡µç«¯æˆ–ç”¨äºŽè®­ç»ƒåŸºç¡€æ¨¡åž‹ã€‚
- **æ–°é—»ç«™ç‚¹**ï¼šé‡‡é›†å‰æ£€æŸ¥ robots.txtï¼Œå¿…è¦æ—¶ä¸Žç«™ç‚¹ç­¾è®¢æˆ–ç”³è¯·é¢å¤–æŽˆæƒã€‚
- **æ•°æ®åˆ é™¤**ï¼šæ”¶åˆ° X Compliance é€šçŸ¥æˆ–ç‰ˆæƒè¦æ±‚æ—¶ï¼Œéœ€åŒæ­¥åˆ é™¤æœ¬åœ°å­˜æ¡£ã€‚

---

å¦‚éœ€å°†ä¸Šè¿°æµç¨‹æ‰“åŒ…æˆæœ€å°å¯è¿è¡Œæ¨¡æ¿ï¼ˆå«é…ç½®ç¤ºä¾‹ã€è°ƒåº¦è„šæœ¬æˆ–å¯è§†åŒ–ç¤ºä¾‹ï¼‰ï¼Œæ¬¢è¿Žç»§ç»­å‘ŠçŸ¥éœ€æ±‚ã€‚

---

## é“¾ä¸Š DEX æ•°æ®é‡‡é›†æµç¨‹ï¼ˆUniswap v2 / v3 ç¤ºä¾‹ï¼‰

| æ­¥éª¤ | å‘½ä»¤ | è¯´æ˜Ž |
| --- | --- | --- |
| 1 | ç¼–è¾‘ `conf/dex.yaml` å¹¶åœ¨ `.env` ä¸­è®¾ç½® `RPC_URL_ETHEREUM`ï¼ˆHTTPï¼‰å’Œå¯é€‰ `RPC_URL_ETHEREUM_WS` | æŒ‡å®šé“¾ IDã€å·¥åŽ‚åœ°å€ã€èµ·å§‹åŒºå—ã€ç¡®è®¤æ•°ç­‰ |
| 2 | `python scripts/dex_indexer.py --config conf/dex.yaml --output data/raw/dex` | é€šè¿‡ `eth_getLogs` åˆ†å—å›žè¡¥ pair/pool ä¸Ž swap äº‹ä»¶ï¼Œè‡ªåŠ¨è·³è¿‡ç¼ºçœ RPC |
| 3 | `python scripts/live_subscribe.py --max-events 20` | åŸºäºŽ HTTP è½®è¯¢çš„å®žæ—¶è¡¥æ•°ï¼ˆéœ€äº‹å…ˆç´¢å¼•ç”Ÿæˆ pool åˆ—è¡¨ï¼Œå¯æ›¿æ¢ä¸º WebSocketï¼‰ |
| 4 | `python scripts/build_candles.py --input data/raw/dex/uniswap-v3_swaps.parquet --rule 1min` | å°† swap æ•°æ®èšåˆä¸º 1 åˆ†é’Ÿèœ¡çƒ›ï¼Œè¾“å‡ºåˆ° `data/clean/dex/candles_1T.parquet` |

> è¿è¡Œå‰è¯·åœ¨ `.env` ä¸­è¡¥å…… `RPC_URL_ETHEREUM=...`ï¼Œè‹¥éœ€å¤šé“¾å¯å¤åˆ¶ `dex.yaml` å¹¶è°ƒæ•´ `factory`/`start_block`ã€‚
