# George Soros Pairs Trading Bot - Setup Instructions

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create a new directory
mkdir pairs-trading-bot
cd pairs-trading-bot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. File Structure

Create the following files in your project directory:

```
pairs-trading-bot/
│
├── app.py                    # Main Streamlit application
├── chatgpt_assistant.py      # ChatGPT integration module
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (optional)
```

### 3. OpenAI API Setup (Optional but Recommended)

To enable the ChatGPT assistant functionality:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set up your API key:

**Option A: Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: .env file**
```
OPENAI_API_KEY=your-api-key-here
```

**Option C: Direct in code (not recommended for production)**
```python
# In app.py, modify the assistant initialization
assistant = TradingAssistant(api_key="your-api-key-here")
```

### 4. Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Features

### Stock Pair Selection
- **Predefined Pairs**: Choose from 10 curated stock pairs across different sectors
- **Manual Entry**: Enter any two stock tickers for custom analysis

### Fundamental Analysis
- Real-time company information
- Key metrics: P/E Ratio, Market Cap, Volume, Dividend Yield, Beta
- Sector and industry classification

### Pairs Trading Strategy
- **Cointegration Testing**: Statistical validation of pair relationship
- **Z-Score Analysis**: Entry/exit signal generation
- **Position Management**: Long/short signals with exit conditions

### Performance Metrics
- Total Return & CAGR
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown
- Win Rate & Trade Count
- Average Holding Period

### Visualizations
1. **Spread & Z-Score Chart**: Shows mean reversion opportunities
2. **Price Chart**: Normalized prices with trade signals
3. **P&L Analysis**: Daily and cumulative profit/loss
4. **Trade Log**: Recent trading signals and positions

### AI Trading Assistant
- Explains trading concepts
- Interprets analysis results
- Answers strategy questions
- Powered by ChatGPT (requires API key)

## 🔧 Customization

### Modify Trading Parameters

In `app.py`, adjust these parameters:

```python
# Z-score thresholds
long_threshold = -1.5    # Default: -1.5
short_threshold = 1.5    # Default: 1.5
exit_threshold = 0.5     # Default: 0.5

# Lookback period for z-score
lookback_period = 20     # Default: 20 days
```

### Add New Stock Pairs

```python
PREDEFINED_PAIRS = {
    "Your Category": ["TICK1", "TICK2"],
    # Add more pairs here
}
```

## ⚠️ Important Notes

1. **Market Data**: The app uses Yahoo Finance for real-time data. Some tickers may not be available.

2. **Trading Risks**: This is an educational tool. Real trading involves significant risks. Always:
   - Backtest thoroughly
   - Consider transaction costs
   - Implement proper risk management
   - Never trade with money you can't afford to lose

3. **API Limits**: If using ChatGPT, be aware of OpenAI API rate limits and costs.

## 🐛 Troubleshooting

### Common Issues

1. **"Insufficient data for analysis"**
   - Try a different date range or stock pair
   - Ensure both stocks have trading data for the selected period

2. **ChatGPT not working**
   - Verify your API key is correctly set
   - Check your OpenAI account has available credits
   - Ensure you have internet connectivity

3. **Slow performance**
   - Analysis may take time for large date ranges
   - Consider reducing the analysis period
   - Check your internet connection for data downloads

## 📈 Trading Strategy Explained

The pairs trading strategy implemented here:

1. **Identifies Statistical Relationships**: Uses cointegration testing to find stock pairs that move together
2. **Calculates Spread**: Finds the price relationship using linear regression
3. **Generates Signals**: 
   - Long when z-score < -1.5 (spread too low)
   - Short when z-score > 1.5 (spread too high)
   - Exit when |z-score| < 0.5 (spread normalizes)
4. **Market Neutral**: Profits from relative movements, not market direction

## 🤝 Contributing

Feel free to enhance the bot with:
- Additional technical indicators
- More sophisticated entry/exit rules
- Risk management features
- Portfolio optimization
- Real-time trading connections

## 📚 Resources

- [Pairs Trading: Performance of a Relative-Value Arbitrage Rule](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615)
- [Statistical Arbitrage in the U.S. Equities Market](https://www.math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb071008.pdf)
- [George Soros: The Alchemy of Finance](https://www.amazon.com/Alchemy-Finance-George-Soros/dp/0471445495)

---

*Remember: Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.*
