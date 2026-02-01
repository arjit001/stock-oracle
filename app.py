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

# SMART SYMBOL MAP (Maps common names to Tickers)
STOCK_MAP = {
    "RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
    "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS": "INFY.NS", "ITC": "ITC.NS", "TCS": "TCS.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "APPLE": "AAPL", "TESLA": "TSLA", "GOOGLE": "GOOGL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA",
    "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "SOLANA": "SOL-USD"
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
    /* Green/Red Text for Table */
    .bullish { color: #00CC96; font-weight: bold; }
    .bearish { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT DATA ENGINE
# ==========================================
@st.cache_data(ttl=300)
def get_stock_data(query):
    # 1. Resolve Symbol
    # If user typed "Apple", map to "AAPL". If "Reliance", map to "RELIANCE.NS"
    symbol = STOCK_MAP.get(query.upper(), query.upper())
    
    df = None
    source = ""
    error = None
    
    # 2. ATTEMPT 1: Try Symbol AS IS (Good for AAPL, BTC-USD, ZOMATO.NS)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        if not df.empty:
            source = "Yahoo Finance"
    except:
        pass

    # 3. ATTEMPT 2: Try appending .NS (If 'RELIANCE' failed, try 'RELIANCE.NS')
    if (df is None or df.empty) and not symbol.endswith(".NS") and "-" not in symbol:
        try:
            india_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(india_symbol)
            df = ticker.history(period="2y")
            if not df.empty:
                symbol = india_symbol # Update symbol
                source = "Yahoo Finance (India)"
        except:
            pass

    # 4. ATTEMPT 3: NSE PYTHON (If Yahoo blocked Indian stock)
    # Only try this if it looks like an Indian stock (no hyphen, no .NS suffix needed for NSEPython)
    if (df is None or df.empty):
        clean_sym = symbol.replace(".NS", "")
        # Heuristic: NSE codes are usually uppercase letters, no numbers/hyphens
        if clean_sym.isalpha():
            try:
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
            except:
                pass

    if df is None or df.empty:
        return None, None, f"Could not find data for '{query}'. Try the exact ticker (e.g., RELIANCE.NS, AAPL)."

    # 5. DETAILED ANALYSIS
    # Currency
    currency = "₹" if ".NS" in symbol or source == "NSE Direct (Fallback)" else "$"
    
    # Technicals
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)

    # Support/Resistance
    support = df['Low'].tail(50).min()
    resistance = df['High'].tail(50).max()

    # Signals
    last = df.iloc[-1]
    signals = []
    score = 0
    
    # Trend
    if last['Close'] > last['EMA_200']: 
        score += 2; signals.append("Trend: Bullish (>200 EMA)")
    else: 
        score -= 2; signals.append("Trend: Bearish (<200 EMA)")
        
    # Momentum
    if last['RSI'] < 30: score += 1; signals.append("RSI: Oversold (Buy Dip)")
    elif last['RSI'] > 70: score -= 1; signals.append("RSI: Overbought (Caution)")
    
    # Verdict
    if score >= 2: verdict = "STRONG BUY 🚀"
    elif score >= 1: verdict = "BUY 📈"
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
    upside = ((target - fund['price']) / fund['price']) * 100
    
    html = f"""
    <html><body style="font-family:sans-serif; padding:30px; color:#333;">
        <div style="border-bottom:2px solid #333; padding-bottom:10px;">
            <h1 style="margin:0;">{fund['name']} Report</h1>
            <p style="margin:5px 0; color:#666;">Generated by StockOracle | Source: {fund['source']}</p>
        </div>
        <div style="background:{'#e6fffa' if 'BUY' in fund['verdict'] else '#fff5f5'}; padding:20px; border-left:5px solid #333; margin:20px 0;">
            <h2 style="margin:0;">VERDICT: {fund['verdict']}</h2>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Current Price:</b> {fund['currency']} {fund['price']:.2f}</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Target (60D):</b> {fund['currency']} {target:.2f}</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Potential Gain:</b> {upside:.2f}%</div>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px;"><b>Support:</b> {fund['support']:.2f}</div>
        </div>
        <h3>Analysis</h3>
        <ul>{''.join([f'<li>{s}</li>' for s in fund['signals']])}</ul>
    </body></html>
    """
    return html

# ==========================================
# 4. UI SETUP
# ==========================================
with st.sidebar:
    st.title("📊 Settings")
    
    # QUICK SELECTOR
    quick_select = st.selectbox("Quick Select:", ["Select...", "ZOMATO", "RELIANCE", "TATA MOTORS", "APPLE", "TESLA", "BITCOIN"])
    if quick_select != "Select...":
        st.session_state.selected_stock = quick_select
    
    st.divider()
    if 'user_tier' not in st.session_state: st.session_state.user_tier = "Guest"
    st.info(f"User: **{st.session_state.user_tier}**")
    
    if st.session_state.user_tier == "Guest":
        if st.button("💎 Unlock Pro"): st.session_state.user_tier = "Pro"; st.rerun()

st.title("📈 StockOracle: Master Edition")

if st.session_state.user_tier == "Guest":
    st.markdown("""
    <div class="ad-banner">
        <span style="color:#FFD700; font-weight:bold;">📢 OPEN FREE DEMAT ACCOUNT</span><br>
        <span style="color:#ccc; font-size:0.9em;">Zero Brokerage for 30 Days • Sign Up Now</span>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Analysis", "📊 Comparison Table (Pro)"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab1:
    default = st.session_state.get('selected_stock', 'RELIANCE')
    query = st.text_input("Enter Symbol (e.g. Zomato, AAPL, BTC-USD):", value=default)

    if st.button("🚀 Analyze", type="primary"):
        with st.spinner("Crunching Numbers..."):
            df, fund, error = get_stock_data(query)
            
            if error:
                st.error(error)
            else:
                # METRICS
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"{fund['currency']} {fund['price']:.2f}", f"{fund['change']:.2f}%")
                target_price = predict_60_days(df)[1][-1]
                upside = ((target_price - fund['price']) / fund['price']) * 100
                c2.metric("Target (60D)", f"{fund['currency']} {target_price:.0f}", f"{upside:.1f}%")
                c3.metric("Support", f"{fund['support']:.0f}")
                c4.metric("Resistance", f"{fund['resistance']:.0f}")

                # CHART
                f_dates, f_prices = predict_60_days(df)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='Forecast', line=dict(color='#ab47bc', width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='Trend (200 EMA)', line=dict(color='blue', width=1)))
                fig.update_layout(height=500, template="plotly_dark", title=f"{fund['name']} ({fund['source']})", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # REPORT
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🤖 AI Verdict")
                    st.markdown(f"### {fund['verdict']}")
                    for s in fund['signals']: st.write(f"• {s}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c_right:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("📄 Download Report")
                    html = create_html_report(fund, df, (f_dates, f_prices))
                    b64 = base64.b64encode(html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{fund["name"]}_Report.html" style="text-decoration:none; color:white; background:#00CC96; padding:10px 20px; border-radius:5px; font-weight:bold;">📥 Download HTML Report</a>'
                    
                    if st.session_state.user_tier == "Pro":
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("🔒 Locked (Guest)")
                        st.button("Unlock")
                    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: COMPARISON TABLE (Requested Feature) ---
with tab2:
    st.subheader("📊 Multi-Asset Comparison")
    
    # 1. Selection
    options = ["RELIANCE", "TATA MOTORS", "ZOMATO", "APPLE", "TESLA", "BITCOIN", "HDFC BANK", "INFOSYS"]
    selected = st.multiselect("Select stocks to compare:", options, default=["RELIANCE", "APPLE"])
    
    if st.button("Generate Comparison Table"):
        if not selected:
            st.error("Select at least one stock.")
        else:
            table_data = []
            progress = st.progress(0)
            
            for i, stock in enumerate(selected):
                d, f, e = get_stock_data(stock)
                if d is not None:
                    # Calculate Upside
                    target = predict_60_days(d)[1][-1]
                    upside = ((target - f['price']) / f['price']) * 100
                    
                    # Row Data
                    table_data.append({
                        "Symbol": f['name'],
                        "Price": f"{f['currency']} {f['price']:.2f}",
                        "Verdict": f['verdict'],
                        "Target (60D)": f"{f['currency']} {target:.2f}",
                        "Potential Gain": f"{upside:.2f}%",
                        "Source": f['source']
                    })
                progress.progress((i + 1) / len(selected))
            
            # 2. Display Table
            if table_data:
                res_df = pd.DataFrame(table_data)
                
                # Apply Color Formatting
                def highlight_verdict(val):
                    color = '#00CC96' if 'BUY' in val else '#FF4B4B' if 'SELL' in val else 'white'
                    return f'color: {color}; font-weight: bold'
                
                def highlight_gain(val):
                    val_num = float(val.replace('%', ''))
                    color = '#00CC96' if val_num > 0 else '#FF4B4B'
                    return f'color: {color}'

                st.dataframe(
                    res_df.style.map(highlight_verdict, subset=['Verdict'])
                                .map(highlight_gain, subset=['Potential Gain']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.error("No data fetched.")
