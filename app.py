import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import yfinance as yf
import requests
from nsepython import equity_history

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Pro", page_icon="📈")

# GLOBAL STOCK SESSION STATE (For Comparison)
if 'comp_stocks' not in st.session_state:
    st.session_state.comp_stocks = ["RELIANCE.NS", "AAPL.US", "BTC.CR"]

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00CC96; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .bullish { color: #00CC96; font-weight: bold; }
    .bearish { color: #FF4B4B; font-weight: bold; }
    .tag-container { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
    .stock-tag { background: #333; padding: 5px 10px; border-radius: 5px; border: 1px solid #555; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SUFFIX PARSER (The Magic Logic)
# ==========================================
def parse_symbol(user_input):
    s = user_input.upper().strip()
    
    # 1. Check Explicit Suffixes
    if s.endswith(".US"):
        return s.replace(".US", "") # Yahoo uses 'AAPL', not 'AAPL.US'
    elif s.endswith(".CR"):
        clean = s.replace(".CR", "")
        return f"{clean}-USD" # Convert 'BTC.CR' -> 'BTC-USD'
    elif s.endswith(".NS"):
        return s # Keep as is
        
    # 2. Smart Dictionary (If no suffix provided)
    MAP = {
        "RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", 
        "TATAMOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
        "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", 
        "HDFC BANK": "HDFCBANK.NS", "INFOSYS": "INFY.NS",
        "APPLE": "AAPL", "TESLA": "TSLA", "BITCOIN": "BTC-USD"
    }
    return MAP.get(s, s) # Default to input if not in map

# ==========================================
# 3. DATA ENGINE
# ==========================================
@st.cache_data(ttl=300)
def get_stock_data(query):
    symbol = parse_symbol(query)
    
    # Determine Source Type
    is_india = symbol.endswith(".NS")
    
    df = None
    source = ""
    error = None

    # ATTEMPT 1: YAHOO FINANCE
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        if not df.empty:
            source = "Yahoo Finance"
    except: pass

    # ATTEMPT 2: NSE PYTHON (India Fallback)
    if (df is None or df.empty) and is_india:
        try:
            clean = symbol.replace(".NS", "")
            series = equity_history(clean, "EQ", "01-01-2024", datetime.now().strftime("%d-%m-%Y"))
            if series and len(series) > 5:
                df = pd.DataFrame(series)
                df = df.rename(columns={'CH_TIMESTAMP': 'Date', 'CH_CLOSING_PRICE': 'Close', 'CH_OPENING_PRICE': 'Open', 'CH_TRADE_HIGH_PRICE': 'High', 'CH_TRADE_LOW_PRICE': 'Low', 'CH_TOT_TRADED_QTY': 'Volume'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                df = df.astype(float)
                source = "NSE Direct"
        except: pass

    if df is None or df.empty:
        return None, None, f"Not Found: {query}"

    # ANALYSIS
    currency = "₹" if is_india or source == "NSE Direct" else "$"
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    last = df.iloc[-1]
    score = 0
    signals = []
    
    # Trend
    if last['Close'] > last['EMA_200']: score += 2
    else: score -= 2
    
    # RSI
    if last['RSI'] < 30: score += 1
    elif last['RSI'] > 70: score -= 1

    if score >= 2: verdict = "STRONG BUY 🚀"
    elif score <= -2: verdict = "SELL 🔻"
    else: verdict = "HOLD ✋"

    # Prediction
    df_reg = df.reset_index()
    df_reg['ordinal'] = df_reg[df_reg.columns[0]].apply(lambda x: x.toordinal())
    model = LinearRegression().fit(df_reg[['ordinal']], df_reg['Close'])
    future_date = df_reg[df_reg.columns[0]].iloc[-1] + timedelta(days=60)
    target = model.predict([[future_date.toordinal()]])[0]
    
    upside = ((target - last['Close']) / last['Close']) * 100

    fund = {
        "name": symbol,
        "price": last['Close'],
        "currency": currency,
        "verdict": verdict,
        "target": target,
        "upside": upside,
        "source": source
    }
    return df, fund, None

# ==========================================
# 4. UI SETUP
# ==========================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("User Status: **Pro**") # Simulating Pro for now

st.title("📈 StockOracle: Master Edition")

# --- SUFFIX GUIDE ---
st.info("💡 **Suffix Guide:** India=` .ns ` (Zomato.ns) | USA=` .us ` (Apple.us) | Crypto=` .cr ` (Btc.cr)")

tab1, tab2 = st.tabs(["🔍 Single Analysis", "📊 Comparison Table"])

# --- TAB 1: SINGLE ANALYSIS ---
with tab1:
    q = st.text_input("Enter Symbol:", "ZOMATO.NS")
    if st.button("Analyze"):
        df, fund, err = get_stock_data(q)
        if err: st.error(err)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{fund['currency']} {fund['price']:.2f}")
            c2.metric("Target (60D)", f"{fund['currency']} {fund['target']:.2f}", f"{fund['upside']:.2f}%")
            c3.metric("Verdict", fund['verdict'])
            
            # Chart
            fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig.update_layout(height=400, template="plotly_dark", title=f"{fund['name']} Chart")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: COMPARISON TABLE ---
with tab2:
    st.subheader("📊 Portfolio Comparison")
    
    # 1. INPUT AREA
    c_in, c_btn = st.columns([3, 1])
    with c_in:
        new_stock = st.text_input("Add Stock (e.g., TATAMOTORS.NS, TSLA.US, ETH.CR):", key="comp_input")
    with c_btn:
        if st.button("Add to List"):
            if new_stock:
                # Add to session state list if not exists
                clean_stock = new_stock.upper().strip()
                if clean_stock not in st.session_state.comp_stocks:
                    st.session_state.comp_stocks.append(clean_stock)

    # 2. SHOW SELECTED TAGS
    st.write("Selected Stocks:")
    tags_cols = st.columns(8)
    for i, s in enumerate(st.session_state.comp_stocks):
        # Create a mini remove button for each
        if tags_cols[i % 8].button(f"❌ {s}", key=f"rem_{s}"):
            st.session_state.comp_stocks.remove(s)
            st.rerun()

    st.divider()

    # 3. GENERATE TABLE
    if st.button("🚀 Generate Comparison Table"):
        if not st.session_state.comp_stocks:
            st.warning("Please add some stocks first.")
        else:
            rows = []
            progress = st.progress(0)
            total = len(st.session_state.comp_stocks)
            
            for i, ticker in enumerate(st.session_state.comp_stocks):
                _, f, err = get_stock_data(ticker)
                
                if f:
                    rows.append({
                        "Symbol": f['name'],
                        "Price": f"{f['currency']} {f['price']:.2f}",
                        "Verdict": f['verdict'],
                        "Target (60D)": f"{f['currency']} {f['target']:.2f}",
                        "Potential Gain": f"{f['upside']:.2f}%",
                        "Source": f['source']
                    })
                progress.progress((i + 1) / total)
            
            if rows:
                res_df = pd.DataFrame(rows)
                
                # COLOR STYLING
                def style_verdict(v):
                    color = '#00CC96' if 'BUY' in v else '#FF4B4B' if 'SELL' in v else 'white'
                    return f'color: {color}; font-weight: bold'
                
                def style_gain(v):
                    val = float(v.replace('%', ''))
                    color = '#00CC96' if val > 0 else '#FF4B4B'
                    return f'color: {color}'

                st.dataframe(
                    res_df.style.map(style_verdict, subset=['Verdict'])
                                .map(style_gain, subset=['Potential Gain']),
                    use_container_width=True,
                    height=500
                )
            else:
                st.error("Could not fetch data for selected stocks.")
