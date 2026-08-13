# Market Data Monitor

Automated scanner for active Binance Global Spot USDT pairs using closed candles only.

Timeframes: 15m, 1h and 4h.

A Telegram alert is sent only when every required condition for the timeframe is true on the same closed candle. Duplicate alerts for the same symbol, timeframe and candle are prevented.

Required repository secrets:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
