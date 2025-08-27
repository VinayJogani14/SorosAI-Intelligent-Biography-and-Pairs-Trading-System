import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv

# Import ChatGPT assistant
try:
    from chatgpt_assistant import TradingAssistant
    CHATGPT_AVAILABLE = True
except ImportError:
    CHATGPT_AVAILABLE = False

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="George Soros Pairs Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Card-like containers */
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] > div {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    [data-testid="metric-container"] label {
        font-size: 14px;
        color: #888;
        margin-bottom: 5px;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 20px;
        font-weight: bold;
        color: #fff;
    }
    
    /* Success/Warning boxes */
    .stAlert {
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        background-color: #262730;
        border-radius: 8px;
        border: 1px solid #333;
        color: #fff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Chat containers */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    
    .user-message {
        background-color: #2b3340;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #1e1e1e;
        margin-right: 20%;
    }
    
    /* Section dividers */
    .section-divider {
        height: 2px;
        background-color: #333;
        margin: 30px 0;
    }
    
    /* Loading skeleton */
    .loading-skeleton {
        background: linear-gradient(90deg, #1e1e1e 25%, #2b2b2b 50%, #1e1e1e 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        height: 60px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("# 🏦 George Soros Pairs Trading Bot")
st.markdown("*Implementing market-neutral strategies with statistical arbitrage*")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_metrics' not in st.session_state:
    st.session_state.current_metrics = None
if 'current_stock1' not in st.session_state:
    st.session_state.current_stock1 = None
if 'current_stock2' not in st.session_state:
    st.session_state.current_stock2 = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Initialize ChatGPT assistant
if 'assistant' not in st.session_state:
    if CHATGPT_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                st.session_state.assistant = TradingAssistant(api_key=api_key)
                st.session_state.chatgpt_available = True
            except Exception as e:
                st.session_state.chatgpt_available = False
                st.sidebar.warning("ChatGPT unavailable. Set OPENAI_API_KEY to enable.")
        else:
            st.session_state.chatgpt_available = False
    else:
        st.session_state.chatgpt_available = False

# Predefined stock pairs
PREDEFINED_PAIRS = {
    "Energy Giants": ["XOM", "CVX"],
    "Banking Titans": ["JPM", "BAC"],
    "Tech Leaders": ["MSFT", "GOOGL"],
    "Retail Kings": ["WMT", "TGT"],
    "Airlines": ["DAL", "UAL"],
    "Semiconductors": ["NVDA", "AMD"],
    "Pharma": ["PFE", "JNJ"],
    "Streaming": ["NFLX", "DIS"],
    "Payment Processing": ["V", "MA"],
    "E-commerce": ["AMZN", "EBAY"]
}

# Sidebar configuration
with st.sidebar:
    st.header("📊 Stock Pair Selection")
    
    selection_method = st.radio(
        "Choose selection method:",
        ["Predefined Pairs", "Manual Entry"],
        label_visibility="visible"
    )
    
    if selection_method == "Predefined Pairs":
        selected_pair_name = st.selectbox(
            "Select a stock pair:",
            list(PREDEFINED_PAIRS.keys())
        )
        stock1, stock2 = PREDEFINED_PAIRS[selected_pair_name]
    else:
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.text_input("First ticker:", "AAPL").upper()
        with col2:
            stock2 = st.text_input("Second ticker:", "MSFT").upper()
        st.caption("💡 Popular pairs: AAPL/MSFT, COKE/PEP, GM/F")
    
    st.markdown("---")
    
    st.header("📅 Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365),
            label_visibility="visible"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            label_visibility="visible"
        )
    
    st.markdown("---")
    
    # Analysis buttons
    if st.button("🔍 Analyze Pair", type="primary", use_container_width=True):
        st.session_state.analysis_run = True
        st.session_state.current_stock1 = stock1
        st.session_state.current_stock2 = stock2
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Function to get stock fundamentals
@st.cache_data
def get_stock_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Fix dividend yield calculation
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            # Check if the value seems to be already in percentage form (> 1)
            if dividend_yield > 1:
                dividend_yield_str = f"{dividend_yield:.2f}%"
            else:
                dividend_yield_str = f"{dividend_yield*100:.2f}%"
        else:
            dividend_yield_str = "0%"
        
        fundamentals = {
            "Company": info.get("longName", ticker),
            "Sector": info.get("sector", "N/A"),
            "P/E Ratio": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
            "Market Cap": f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A",
            "Dividend Yield": dividend_yield_str,
            "Beta": f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"
        }
        return fundamentals
    except:
        return {
            "Company": ticker,
            "Sector": "Error loading",
            "P/E Ratio": "N/A",
            "Market Cap": "N/A",
            "Dividend Yield": "N/A",
            "Beta": "N/A"
        }

# Function to perform pairs trading analysis
@st.cache_data(show_spinner=False)
def pairs_trading_analysis(stock1, stock2, start_date, end_date):
    try:
        # Download data
        data1 = yf.download(stock1, start=start_date, end=end_date, auto_adjust=True, progress=False)
        data2 = yf.download(stock2, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if data1.empty or data2.empty:
            return None, "One or both stocks returned no data. Please check the ticker symbols."
        
        # Fix: Extract Close prices and align indices
        close1 = data1['Close']
        close2 = data2['Close']
        
        # Create DataFrame with aligned dates
        data = pd.DataFrame(index=close1.index)
        data[stock1] = close1
        data[stock2] = close2
        
        # Drop any NaN values
        data = data.dropna()
        
    except Exception as e:
        return None, f"Error downloading data: {str(e)}"
    
    if data.empty or len(data) < 30:
        return None, "Insufficient data for analysis (need at least 30 days)"
    
    # Cointegration test
    score, pvalue, _ = coint(data[stock1], data[stock2])
    
    # Calculate spread using OLS
    X = sm.add_constant(data[stock1])
    y = data[stock2]
    model = sm.OLS(y, X).fit()
    
    # Calculate spread
    data['spread'] = y - model.predict(X)
    
    # Calculate z-score
    data['z_score'] = (data['spread'] - data['spread'].rolling(20).mean()) / data['spread'].rolling(20).std()
    
    # Generate signals
    data['long_signal'] = data['z_score'] < -1.5
    data['short_signal'] = data['z_score'] > 1.5
    data['exit_signal'] = abs(data['z_score']) < 0.5
    
    # Calculate returns
    data['returns_stock1'] = data[stock1].pct_change()
    data['returns_stock2'] = data[stock2].pct_change()
    
    # Calculate PnL
    position = 0
    pnl = []
    positions = []
    
    for i in range(1, len(data)):
        if data['long_signal'].iloc[i] and position == 0:
            position = 1
        elif data['short_signal'].iloc[i] and position == 0:
            position = -1
        elif data['exit_signal'].iloc[i] and position != 0:
            position = 0
        
        positions.append(position)
        
        if position == 1:
            daily_pnl = data['returns_stock2'].iloc[i] - data['returns_stock1'].iloc[i]
        elif position == -1:
            daily_pnl = data['returns_stock1'].iloc[i] - data['returns_stock2'].iloc[i]
        else:
            daily_pnl = 0
        
        pnl.append(daily_pnl)
    
    data['position'] = [0] + positions
    data['pnl'] = [0] + pnl
    data['cumulative_pnl'] = data['pnl'].cumsum()
    
    # Calculate metrics
    total_return = data['cumulative_pnl'].iloc[-1]
    
    # Sharpe Ratio (annualized)
    if data['pnl'].std() > 0:
        sharpe_ratio = (data['pnl'].mean() * 252) / (data['pnl'].std() * np.sqrt(252))
    else:
        sharpe_ratio = 0
    
    # Max Drawdown
    cumulative = (1 + data['pnl']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Trade statistics
    trades = data['position'].diff() != 0
    trade_count = trades.sum() // 2
    
    winning_trades = data[data['pnl'] > 0]['pnl'].count()
    total_trades = data[data['pnl'] != 0]['pnl'].count()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    metrics = {
        "Cointegration p-value": pvalue,
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trades": int(trade_count),
        "Win Rate": win_rate,
        "Beta (Hedge Ratio)": model.params[1]
    }
    
    return data, metrics

# Main content area
if st.session_state.analysis_run:
    # Stock information section
    st.markdown("## 📊 Stock Analysis")
    
    col1, col2, col3 = st.columns([1, 0.1, 1])
    
    # Stock 1 Fundamentals
    with col1:
        st.markdown(f"### {stock1}")
        with st.spinner(f"Loading {stock1} data..."):
            fund1 = get_stock_fundamentals(stock1)
        
        for key, value in fund1.items():
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**{key}:**")
            with col_b:
                st.markdown(f"{value}")
    
    # Stock 2 Fundamentals
    with col3:
        st.markdown(f"### {stock2}")
        with st.spinner(f"Loading {stock2} data..."):
            fund2 = get_stock_fundamentals(stock2)
        
        for key, value in fund2.items():
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**{key}:**")
            with col_b:
                st.markdown(f"{value}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Pairs Trading Analysis
    st.markdown("## 🎯 Pairs Trading Analysis")
    
    with st.spinner(f"Analyzing {stock1} and {stock2} pair..."):
        data, metrics = pairs_trading_analysis(stock1, stock2, start_date, end_date)
        
    if isinstance(metrics, str):  # Error message
        st.error(metrics)
    else:
        st.session_state.analysis_data = data
        st.session_state.current_metrics = metrics
        
        # Cointegration status
        col1, col2 = st.columns([3, 1])
        with col1:
            if metrics["Cointegration p-value"] < 0.05:
                st.success(f"✅ Stocks are cointegrated (p-value: {metrics['Cointegration p-value']:.4f}) - Suitable for pairs trading")
            else:
                st.warning(f"⚠️ Stocks may not be cointegrated (p-value: {metrics['Cointegration p-value']:.4f}) - Trade with caution")
        
        # Strategy Performance Metrics
        st.markdown("### 📈 Strategy Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['Total Return']*100:.2f}%",
                delta=f"{metrics['Total Return']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']:.2f}",
                delta="Good" if metrics['Sharpe Ratio'] > 1 else "Poor"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['Max Drawdown']*100:.2f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['Win Rate']*100:.1f}%",
                delta=None
            )
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics['Number of Trades'])
        
        with col2:
            st.metric("Hedge Ratio", f"{metrics['Beta (Hedge Ratio)']:.4f}")
        
        with col3:
            days_analyzed = len(data)
            st.metric("Days Analyzed", days_analyzed)
        
        with col4:
            current_position = "Long" if data['position'].iloc[-1] == 1 else "Short" if data['position'].iloc[-1] == -1 else "Neutral"
            st.metric("Current Position", current_position)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("## 📊 Trading Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Spread & Z-Score", "Price Chart", "P&L Analysis", "Trade Log"])
        
        with tab1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            
            # Spread
            ax1.plot(data.index, data['spread'], label='Spread', color='#00d4ff', alpha=0.8, linewidth=1.5)
            ax1.axhline(data['spread'].mean(), color='white', linestyle='--', alpha=0.5, label='Mean')
            ax1.fill_between(data.index, data['spread'].mean() - data['spread'].std(), 
                           data['spread'].mean() + data['spread'].std(), alpha=0.1, color='gray')
            ax1.set_ylabel('Spread', color='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.2)
            ax1.set_title(f'Spread between {stock1} and {stock2}', color='white', fontsize=14)
            ax1.tick_params(colors='white')
            
            # Z-score
            ax2.plot(data.index, data['z_score'], label='Z-Score', color='#ff6b6b', alpha=0.8, linewidth=1.5)
            ax2.axhline(1.5, color='#ff4757', linestyle='--', label='Short Signal')
            ax2.axhline(-1.5, color='#00d4ff', linestyle='--', label='Long Signal')
            ax2.axhline(0, color='white', linestyle='-', alpha=0.3)
            ax2.fill_between(data.index, -0.5, 0.5, alpha=0.1, color='yellow', label='Exit Zone')
            ax2.set_ylabel('Z-Score', color='white')
            ax2.set_xlabel('Date', color='white')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.2)
            ax2.tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Normalize prices
            norm_stock1 = data[stock1] / data[stock1].iloc[0] * 100
            norm_stock2 = data[stock2] / data[stock2].iloc[0] * 100
            
            ax.plot(data.index, norm_stock1, label=stock1, color='#00d4ff', linewidth=2)
            ax.plot(data.index, norm_stock2, label=stock2, color='#ff6b6b', linewidth=2)
            
            # Trading signals
            long_signals = data[data['long_signal'] & (data['position'].diff() != 0)]
            short_signals = data[data['short_signal'] & (data['position'].diff() != 0)]
            
            for idx in long_signals.index:
                ax.axvline(idx, color='#00d4ff', alpha=0.3, linestyle='--')
            for idx in short_signals.index:
                ax.axvline(idx, color='#ff4757', alpha=0.3, linestyle='--')
            
            ax.set_ylabel('Normalized Price (Base 100)', color='white')
            ax.set_xlabel('Date', color='white')
            ax.legend()
            ax.grid(True, alpha=0.2)
            ax.set_title(f'Price Movement: {stock1} vs {stock2}', color='white', fontsize=14)
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            
            # Daily P&L
            colors = ['#00d4ff' if x > 0 else '#ff4757' for x in data['pnl']]
            ax1.bar(data.index, data['pnl'], alpha=0.7, color=colors, width=1)
            ax1.set_ylabel('Daily P&L', color='white')
            ax1.set_title('Daily Profit/Loss', color='white', fontsize=14)
            ax1.grid(True, alpha=0.2)
            ax1.tick_params(colors='white')
            
            # Cumulative P&L
            ax2.plot(data.index, data['cumulative_pnl'], label='Cumulative P&L', 
                    color='#00d4ff', linewidth=2)
            ax2.fill_between(data.index, 0, data['cumulative_pnl'], 
                           where=data['cumulative_pnl'] >= 0, color='#00d4ff', alpha=0.3)
            ax2.fill_between(data.index, 0, data['cumulative_pnl'], 
                           where=data['cumulative_pnl'] < 0, color='#ff4757', alpha=0.3)
            ax2.set_ylabel('Cumulative P&L', color='white')
            ax2.set_xlabel('Date', color='white')
            ax2.set_title('Cumulative Profit/Loss', color='white', fontsize=14)
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            ax2.tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            # Create trade log
            trade_data = data[['spread', 'z_score', 'position', 'pnl']].copy()
            trade_data['signal'] = 'Hold'
            trade_data.loc[data['long_signal'], 'signal'] = 'Long Entry'
            trade_data.loc[data['short_signal'], 'signal'] = 'Short Entry'
            trade_data.loc[data['exit_signal'], 'signal'] = 'Exit'
            
            # Filter for actual trades
            trades = trade_data[trade_data['signal'] != 'Hold'].tail(20)
            
            if not trades.empty:
                trades_display = trades.copy()
                trades_display['spread'] = trades_display['spread'].round(4)
                trades_display['z_score'] = trades_display['z_score'].round(2)
                trades_display['pnl'] = (trades_display['pnl'] * 100).round(2)
                trades_display = trades_display.rename(columns={'pnl': 'P&L (%)'})
                
                st.dataframe(
                    trades_display,
                    use_container_width=True,
                    hide_index=False
                )
            else:
                st.info("No trades executed in the selected period")

else:
    # Welcome screen when no analysis is run
    st.markdown("## 👋 Welcome to the Pairs Trading Bot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 📊 How it works
        1. Select a stock pair
        2. Choose date range
        3. Click "Analyze Pair"
        4. Review trading signals
        """)
    
    with col2:
        st.success("""
        ### 🎯 Key Features
        - Cointegration testing
        - Z-score analysis
        - Automated signals
        - Performance metrics
        """)
    
    with col3:
        st.warning("""
        ### ⚠️ Remember
        - Past performance ≠ future
        - Use proper risk management
        - This is educational only
        - Always do your research
        """)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ChatGPT Assistant Section
st.markdown("## 🤖 Trading Assistant")

# Quick questions
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Z-Score", use_container_width=True):
        response = "Z-score measures how many standard deviations the spread is from its mean. In pairs trading, extreme z-scores signal trading opportunities."
        st.session_state.chat_history.append({
            "question": "What is Z-Score?",
            "answer": f"[Assistant]\n\n{response}",
            "timestamp": datetime.now().strftime("%H:%M")
        })

with col2:
    if st.button("🔗 Cointegration", use_container_width=True):
        response = "Cointegration tests whether two stocks have a stable long-term relationship. Unlike correlation, it indicates a genuine economic connection."
        st.session_state.chat_history.append({
            "question": "Explain Cointegration",
            "answer": f"[Assistant]\n\n{response}",
            "timestamp": datetime.now().strftime("%H:%M")
        })

with col3:
    if st.button("📈 Sharpe Ratio", use_container_width=True):
        response = "Sharpe Ratio measures risk-adjusted returns. Higher values indicate better returns per unit of risk. Above 1 is good, above 2 is excellent."
        st.session_state.chat_history.append({
            "question": "What's Sharpe Ratio?",
            "answer": f"[Assistant]\n\n{response}",
            "timestamp": datetime.now().strftime("%H:%M")
        })

with col4:
    if st.button("📉 Drawdown", use_container_width=True):
        response = "Maximum drawdown shows the largest peak-to-trough decline. It represents the worst loss an investor would have experienced."
        st.session_state.chat_history.append({
            "question": "About Drawdown",
            "answer": f"[Assistant]\n\n{response}",
            "timestamp": datetime.now().strftime("%H:%M")
        })

# Chat input
col1, col2 = st.columns([5, 1])

with col1:
    user_question = st.text_input(
        "Ask the Trading Assistant:",
        placeholder="e.g., Why did the strategy go long? How can I improve the Sharpe ratio?",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and user_question:
    # Add question and response to chat history
    if st.session_state.get('chatgpt_available') and hasattr(st.session_state, 'assistant'):
        context = ""
        if st.session_state.current_metrics and st.session_state.current_stock1 and st.session_state.current_stock2:
            context = st.session_state.assistant.get_trading_context(
                st.session_state.current_stock1, 
                st.session_state.current_stock2, 
                st.session_state.current_metrics
            )
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.assistant.answer_question(user_question, context)
                response_text = f"[ChatGPT]\n\n{response}"
            except Exception as e:
                response_text = f"Error: {str(e)}"
    else:
        response_text = f"""[Demo Response]

Based on pairs trading principles, I can help explain various aspects of the strategy. 
To enable real AI responses, please set up your OpenAI API key in the .env file."""

    st.session_state.chat_history.append({
        "question": user_question,
        "answer": response_text,
        "timestamp": datetime.now().strftime("%H:%M")
    })

# Display chat history
if st.session_state.chat_history:
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
        st.markdown(f'<div class="chat-message user-message">👤 <b>{chat["timestamp"]}</b><br>{chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message assistant-message">🤖 {chat["answer"]}</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
    <small>George Soros Pairs Trading Bot | Educational Purpose Only | Not Financial Advice</small>
    </div>
    """,
    unsafe_allow_html=True
)