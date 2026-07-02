# ⚡ AI Trading Analyst — Global & Indian Stock Markets

> **Fully offline, professional-grade AI trading research platform & autonomous paper trader.**  
> **Created by: Ansh Anand**
> Zero cloud dependency • ₹0 running cost • Simulation mode only by default • Local AI self-learning

---

## 🌍 Global Markets & Timezones Support
- **Universal Stock Resolver**: Supports Indian markets (`RELIANCE.NS`, `TCS.NS`) as well as global exchanges (US NASDAQ/NYSE: `AAPL`, `MSFT`; London: `BP.L`; Toronto: `SHOP.TO`) natively.
- **IST (Indian Standard Time) Integration**: Naive dates are correctly localized as UTC before unix epoch conversion, ensuring TradingView lightweight charts render in your local time zone (IST, UTC+5:30) correctly.
- **SQLite Lock-Retry Self-Healing**: Includes atomic transaction locks and exponential backoff retry algorithms to eliminate SQLite concurrency issues during heavy multi-threaded scanning.

---

## 🚀 Quick Start

### 1. Requirements
- **Python 3.10+**
- **Windows 10/11** (also works on Linux/macOS)
- 4GB+ RAM (8GB recommended for AI models)

### 2. Install
```bash
git clone <repo> && cd tradebot
pip install -r requirements.txt
```

### 3. Launch
```bash
# Option A: Double-click START.bat  (Windows)
# Option B: Command line
python launch.py
```

The browser opens automatically at `http://127.0.0.1:8765`

---

## 🤖 AI Chat (Optional)

Install [Ollama](https://ollama.ai) for local AI-powered market analysis:
```bash
# After installing Ollama:
ollama pull mistral        # 4GB — best balance
ollama pull llama3         # 5GB — most capable
ollama pull phi3           # 2GB — lightweight option
```

The app auto-detects Ollama. If unavailable, rule-based analysis is used instead.

---

## 🏗️ Architecture

```
tradebot/
├── app/
│   ├── main.py              # FastAPI server
│   ├── config.py            # All settings
│   ├── db/                  # SQLite + SQLAlchemy models
│   ├── data/                # yfinance historical data
│   ├── core/
│   │   ├── analysis/        # 20+ indicators, SMC, patterns
│   │   ├── strategies/      # 14 strategy plugins + voting engine
│   │   ├── risk/            # Risk manager + circuit breaker
│   │   ├── simulation/      # Virtual portfolio (paper trading)
│   │   └── scanner/         # Parallel market scanner
│   └── ai/                  # Ollama client + self-learning
├── frontend/
│   └── index.html           # Complete web UI
├── launch.py                # Entry point
└── START.bat                # Windows launcher
```

---

## 📊 Features

| Feature | Status |
|---|---|
| NSE/BSE Cash Market Scanner | ✅ |
| 20+ Technical Indicators | ✅ |
| Smart Money Concepts (SMC) | ✅ |
| 14 Strategy Plugins | ✅ |
| Weighted Voting Engine | ✅ |
| Real Probability Engine | ✅ |
| Virtual Portfolio (Paper Trading) | ✅ |
| Partial Exit (T1/T2/T3) | ✅ |
| Self-Learning (Strategy Weights) | ✅ |
| Local AI Chat (Ollama) | ✅ |
| TradingView-style Charts | ✅ |
| Backtesting Engine | ✅ |
| Circuit Breaker / Risk Manager | ✅ |
| Live Broker Integration | 🔒 Disabled by design |

---

## ⚠️ Important Disclaimers

1. **Simulation Only**: All trades are virtual by default. No real money at risk.
2. **No Guarantees**: AI signals are probabilistic, not guarantees of profit.
3. **Educational Use**: Designed for research and learning, not financial advice.
4. **Data Source**: Historical data from Yahoo Finance (free). NSE real-time data requires a paid data vendor for live use.

---

## 🔧 Configuration

Edit `.env` or `app/config.py`:

```env
VIRTUAL_INITIAL_BALANCE=100000
SCANNER_INTERVAL_SECONDS=60
MIN_CONFIDENCE_THRESHOLD=0.70
MIN_STRATEGY_AGREEMENT=3
MAX_DAILY_LOSS_PCT=2.0
OLLAMA_DEFAULT_MODEL=mistral
```

---

## 📈 Strategy Plugins

| # | Strategy | Category |
|---|---|---|
| 1 | Trend Following (EMA) | Trend |
| 2 | Breakout (Donchian) | Breakout |
| 3 | Pullback to EMA/VWAP | Trend |
| 4 | Mean Reversion (BB) | Mean Reversion |
| 5 | Opening Range Breakout | Breakout |
| 6 | VWAP Reversal | Mean Reversion |
| 7 | SuperTrend + EMA | Trend |
| 8 | RSI Divergence | Reversal |
| 9 | MACD Momentum | Momentum |
| 10 | Smart Money Concepts | Institutional |
| 11 | Price Action (Patterns) | Reversal |
| 12 | Support/Resistance Bounce | Reversal |
| 13 | Gap Trading | Momentum |
| 14 | Volume Breakout | Breakout |

---

## 🏛️ API

Local REST API at `http://127.0.0.1:8765/api/`

- `GET /api/status` — System health
- `POST /api/analyze` — Analyze a stock
- `GET /api/scanner/results` — Scanner results
- `POST /api/scanner/run` — Trigger scan
- `GET /api/portfolio/stats` — Portfolio stats
- `POST /api/portfolio/order` — Place simulated order
- `GET /api/strategies` — All strategies
- `POST /api/chat` — AI chat

Full API docs at `http://127.0.0.1:8765/docs`

---

*Created by **Ansh Anand** with ❤️. ₹0 running cost, 100% offline.*
