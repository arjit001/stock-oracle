import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import yfinance as yf
import requests
from textblob import TextBlob
import base64
from nsepython import equity_history

# ==========================================
# 1. CONFIGURATION & DICTIONARY
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Master", page_icon="📈")

# SMART SYMBOL DICTIONARY (Helps finding Zomato, Bitcoin, etc.)
STOCK_MAP = {
    "RELIANCE IND": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
    "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS": "INFY.NS", "ITC": "ITC.NS", "TCS": "TCS.NS",
    "APPLE": "AAPL", "TESLA": "TSLA", "GOOGLE": "GOOGL", "MICROSOFT": "MSFT",
    "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "GOLD": "GC=F"
}

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
    .ad-banner {
        background: linear-gradient(90deg, #1a1a1a, #333);
        border: 1px dashed #FFD700;
        padding: 12px;
        text-align: center;
        border-radius: 8px;
        cursor: pointer;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HYBRID DATA ENGINE (Yahoo -> NSE Fallback)
# ==========================================
@st.cache_data(ttl=300)
def get_stock_data(query):
    # 1. Resolve Symbol from Dictionary or Input
    symbol = STOCK_MAP.get(query.upper(), query.upper())
    
    # 2. Determine Attributes
    is_india = ".NS" in symbol or (symbol.isalpha() and len(symbol) < 10 and "-" not in symbol)
    if is_india and not symbol.endswith(".NS"): symbol = f"{symbol}.NS"
    currency = "₹" if is_india else "$"
    
    df = None
    source = ""
    error = None

    # --- ATTEMPT 1: YAHOO FINANCE (Global) ---
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        if not df.empty:
            source = "Yahoo Finance"
    except Exception as e:
        error = str(e)

    # --- ATTEMPT 2: NSE PYTHON (India Only Fallback) ---
    if (df is None or df.empty) and is_india:
        try:
            clean_sym = symbol.replace(".NS", "")
            series = equity_history(clean_sym, "EQ", "01-01-2024", datetime.now().strftime("%d-%m-%Y"))
            if series and len(series) > 5:
                df = pd.DataFrame(series)
                df = df.rename(columns={'CH_TIMESTAMP': 'Date', 'CH_CLOSING_PRICE': 'Close', 
                                      'CH_OPENING_PRICE': 'Open', 'CH_TRADE_HIGH_PRICE': 'High', 
                                      'CH_TRADE_LOW_PRICE': 'Low', 'CH_TOT_TRADED_QTY': 'Volume'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                df = df.astype(float)
                source = "NSE Direct (Fallback)"
        except Exception as e:
            error = str(e)

    if df is None or df.empty:
        return None, None, f"Could not find data for '{query}'. Try the full symbol (e.g. ZOMATO.NS)."

    # 3. DETAILED ANALYSIS
    # Technicals
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)

    # Calculate Support/Resistance (Simple 20-day High/Low)
    support = df['Low'].tail(20).min()
    resistance = df['High'].tail(20).max()

    # Signals
    last = df.iloc[-1]
    signals = []
    score = 0
    
    # Trend
    if last['Close'] > last['EMA_200']: 
        score += 2; signals.append("Trend: Bullish (>200 EMA)")
    else: 
        score -= 2; signals.append("Trend: Bearish (<200 EMA)")
        
    # Golden Cross
    if last['EMA_50'] > last['EMA_200']: signals.append("Pattern: Golden Cross (Strong Buy)")
    
    # Momentum
    if last['RSI'] < 30: score += 1; signals.append("RSI: Oversold (Buy Dip)")
    elif last['RSI'] > 70: score -= 1; signals.append("RSI: Overbought (Caution)")
    
    # Volume Spike
    avg_vol = df['Volume'].tail(20).mean()
    if last['Volume'] > 1.5 * avg_vol: signals.append("Volume: High Buying Pressure")

    # Verdict
    if score >= 2: verdict = "STRONG BUY 🚀"
    elif score <= -2: verdict = "SELL 🔻"
    else: verdict = "HOLD ✋"

    fund = {
        "name": symbol,
        "price": last['Close'],
        "currency": currency,
        "change": (last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100,
        "verdict": verdict,
        "signals": signals,
        "support": support,
        "resistance": resistance,
        "volatility": f"{(df['Close'].pct_change().std()*np.sqrt(252)*100):.1f}%",
        "source": source
    }
    return df, fund, None

# ==========================================
# 3. PREDICTION & REPORT
# ==========================================
def predict_60_days(df):
    df = df.reset_index()
    date_col = df.columns[0]
    df['ordinal'] = df[date_col].apply(lambda x: x.toordinal())
    
    X = df[['ordinal']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = df[date_col].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 61)]
    future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_pred = model.predict(future_ord)
    
    return future_dates, future_pred

def create_html_report(fund, df, prediction_data):
    target = prediction_data[1][-1]
    html = f"""
    <html><body style="font-family:sans-serif; padding:30px; color:#333;">
        <div style="border-bottom:2px solid #333; padding-bottom:10px;">
            <h1 style="margin:0;">{fund['name']} Report</h1>
            <p style="margin:5px 0; color:#666;">Generated by StockOracle | Source: {fund['source']}</p>
        </div>
        <div style="background:{'#e6fffa' if 'BUY' in fund['verdict'] else '#fff5f5'}; padding:20px; border-left:5px solid #333; margin:20px 0;">
            <h2 style="margin:0;">AI VERDICT: {fund['verdict']}</h2>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Current Price:</b> {fund['currency']} {fund['price']:.2f}</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>60-Day Target:</b> {fund['currency']} {target:.2f}</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Support:</b> {fund['support']:.2f}</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Resistance:</b> {fund['resistance']:.2f}</div>
        </div>
        <h3>Technical Signals</h3>
        <ul>{''.join([f'<li>{s}</li>' for s in fund['signals']])}</ul>
    </body></html>
    """
    return html

# ==========================================
# 4. UI: SIDEBAR & MAIN
# ==========================================
with st.sidebar:
    st.title("📊 Settings")
    
    # SMART SEARCH DICTIONARY
    st.subheader("🔍 Quick Find")
    quick_select = st.selectbox("Popular Stocks:", ["Select...", "ZOMATO", "TATA MOTORS", "RELIANCE IND", "BITCOIN", "APPLE"])
    
    if quick_select != "Select...":
        st.session_state.selected_stock = quick_select
    
    st.divider()
    if 'user_tier' not in st.session_state: st.session_state.user_tier = "Guest"
    st.info(f"User: **{st.session_state.user_tier}**")
    
    if st.session_state.user_tier == "Guest":
        if st.button("💎 Unlock Pro"): st.session_state.user_tier = "Pro"; st.rerun()

# MAIN DASHBOARD
st.title("📈 StockOracle: Master Edition")

if st.session_state.user_tier == "Guest":
    st.markdown("""
    <div class="ad-banner">
        <span style="color:#FFD700; font-weight:bold;">📢 OPEN FREE DEMAT ACCOUNT</span><br>
        <span style="color:#ccc; font-size:0.9em;">Zero Brokerage for 30 Days • Sign Up Now</span>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Deep Analysis", "⚖️ Comparison (Max 8)"])

# --- TAB 1: ANALYSIS ---
with tab1:
    # Input Logic (Syncs with Sidebar)
    default = st.session_state.get('selected_stock', 'RELIANCE')
    query = st.text_input("Enter Symbol or Name (e.g., Zomato, Bitcoin):", value=default)

    if st.button("🚀 Analyze", type="primary"):
        with st.spinner("Analyzing Market Data..."):
            df, fund, error = get_stock_data(query)
            
            if error:
                st.error(error)
            else:
                # 1. METRICS
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"{fund['currency']} {fund['price']:.2f}", f"{fund['change']:.2f}%")
                c2.metric("Target (60D)", f"{fund['currency']} {predict_60_days(df)[1][-1]:.2f}")
                c3.metric("Support", f"{fund['support']:.2f}")
                c4.metric("Resistance", f"{fund['resistance']:.2f}")

                # 2. CHART
                f_dates, f_prices = predict_60_days(df)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='Forecast', line=dict(color='#ab47bc', width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='200 EMA (Trend)', line=dict(color='blue', width=1)))
                fig.update_layout(height=500, template="plotly_dark", title=f"{fund['name']} ({fund['source']})", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # 3. DETAILED SIGNALS
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🤖 AI Analysis")
                    st.markdown(f"**Verdict:** {fund['verdict']}")
                    st.markdown(f"**Volatility:** {fund['volatility']}")
                    for s in fund['signals']: st.write(f"• {s}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c_right:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("📄 Report")
                    html = create_html_report(fund, df, (f_dates, f_prices))
                    b64 = base64.b64encode(html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{fund["name"]}_Report.html" style="text-decoration:none; color:white; background:#00CC96; padding:10px 20px; border-radius:5px; font-weight:bold;">📥 Download Full Report</a>'
                    
                    if st.session_state.user_tier == "Pro":
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("🔒 Locked")
                        st.button("Unlock")
                    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: COMPARISON ---
with tab2:
    st.subheader("Compare Assets")
    # Multi-Select with Auto-fill dictionary
    options = list(STOCK_MAP.keys()) + ["RELIANCE.NS", "TCS.NS", "AAPL", "BTC-USD"]
    selected = st.multiselect("Select up to 8 Stocks:", options, default=["RELIANCE IND", "TATA MOTORS"])
    
    if st.button("Run Comparison"):
        if len(selected) > 8:
            st.error("Please select 8 or fewer stocks.")
        else:
            comp_df = pd.DataFrame()
            progress = st.progress(0)
            
            for i, stock_name in enumerate(selected):
                # Use the same smart getter
                d, _, _ = get_stock_data(stock_name)
                if d is not None:
                    # Normalize: (Price / Start_Price - 1) * 100
                    norm = (d['Close'] / d['Close'].iloc[0] - 1) * 100
                    comp_df[stock_name] = norm
                progress.progress((i + 1) / len(selected))
            
            st.line_chart(comp_df)
            st.caption("Chart shows percentage growth (%) over the last 2 years.")
