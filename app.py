import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import yfinance as yf
import requests
from nsepython import equity_history
import base64

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Ultra", page_icon="🚀")

# SMART MAPPING
STOCK_MAP = {
    "RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
    "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS": "INFY.NS", "ITC": "ITC.NS", "TCS": "TCS.NS",
    "APPLE": "AAPL", "TESLA": "TSLA", "GOOGLE": "GOOGL", "MICROSOFT": "MSFT", 
    "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "GOLD": "GC=F"
}

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    /* METRICS */
    div[data-testid="stMetricValue"] { font-size: 22px; color: #00CC96; }
    /* CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    /* PROGRESS BAR FOR WIN RATE */
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #ff4b4b, #fafa6e, #00cc96); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (HYBRID)
# ==========================================
@st.cache_data(ttl=300)
def get_complete_data(query):
    symbol = STOCK_MAP.get(query.upper(), query.upper())
    
    # Suffix Logic
    if ".US" in symbol: symbol = symbol.replace(".US", "")
    if ".CR" in symbol: symbol = symbol.replace(".CR", "-USD")
    if not any(x in symbol for x in [".NS", "-", "="]) and len(symbol) < 10 and symbol.isalpha():
        symbol = f"{symbol}.NS"

    df = None
    source = ""
    info = {}

    # FETCH 1: YAHOO
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        info = ticker.info
        if not df.empty: source = "Yahoo Finance"
    except: pass

    # FETCH 2: NSE PYTHON
    if (df is None or df.empty) and ".NS" in symbol:
        try:
            clean = symbol.replace(".NS", "")
            series = equity_history(clean, "EQ", "01-01-2024", datetime.now().strftime("%d-%m-%Y"))
            if series:
                df = pd.DataFrame(series)
                df = df.rename(columns={'CH_TIMESTAMP': 'Date', 'CH_CLOSING_PRICE': 'Close', 'CH_OPENING_PRICE': 'Open', 'CH_TRADE_HIGH_PRICE': 'High', 'CH_TRADE_LOW_PRICE': 'Low', 'CH_TOT_TRADED_QTY': 'Volume'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index().astype(float)
                source = "NSE Direct"
        except: pass

    if df is None or df.empty: return None, None, f"Not Found: {symbol}"

    # --- ADVANCED CALCULATIONS ---
    # 1. Standard
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    # 2. MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)

    # 3. Bollinger Bands (Volatility)
    bb = ta.bbands(df['Close'], length=20)
    df = pd.concat([df, bb], axis=1)

    # 4. Stochastic Oscillator (Overbought/Oversold)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, stoch], axis=1)

    # 5. ATR (Average True Range for Stop Loss)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 6. VWAP (Intraday Proxy)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # --- INTRADAY PIVOT POINTS (Classic) ---
    last = df.iloc[-1]
    P = (last['High'] + last['Low'] + last['Close']) / 3
    R1 = 2*P - last['Low']
    S1 = 2*P - last['High']

    # --- WIN CHANCE SCORING ---
    score = 0
    total_checks = 6
    
    if last['Close'] > last['EMA_200']: score += 1
    if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 1
    if last['RSI'] < 70 and last['RSI'] > 30: score += 1 # Healthy range
    if last['Close'] > last['VWAP']: score += 1
    if last['STOCHk_14_3_3'] > last['STOCHd_14_3_3']: score += 1 # Stoch Cross
    if last['Close'] > last['BBU_20_2.0']: score -= 1 # Hit upper band (reversal risk)
    
    win_prob = min(max((score / total_checks) * 100, 10), 99)
    
    # Verdict
    if win_prob > 75: verdict = "STRONG BUY 🚀"
    elif win_prob > 55: verdict = "BUY 📈"
    elif win_prob < 30: verdict = "STRONG SELL 📉"
    elif win_prob < 45: verdict = "SELL 🔻"
    else: verdict = "HOLD ✋"

    data_pack = {
        "symbol": symbol,
        "price": last['Close'],
        "currency": "₹" if ".NS" in symbol else "$",
        "change": (last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100,
        "volatility": df['ATR'].iloc[-1],
        "verdict": verdict,
        "win_prob": win_prob,
        "pivot": {"P": P, "R1": R1, "S1": S1},
        "signals": {
            "RSI": last['RSI'],
            "MACD": "Bullish" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "Bearish",
            "Trend": "Up" if last['Close'] > last['EMA_200'] else "Down",
            "Bollinger": "Overbought" if last['Close'] > last['BBU_20_2.0'] else "Oversold" if last['Close'] < last['BBL_20_2.0'] else "Neutral"
        },
        "info": info,
        "source": source
    }
    
    return df, data_pack, None

# ==========================================
# 3. MACHINE LEARNING ENGINE
# ==========================================
def predict_ml(df):
    # Prepare Data for ML
    df = df.copy().dropna()
    df['Target'] = df['Close'].shift(-1) # Predict next day's close
    df = df.dropna()
    
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'EMA_50', 'EMA_200']
    X = df[features]
    y = df['Target']
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict Next 7 Days (Auto-Regressive)
    future_prices = []
    last_row = X.iloc[[-1]].copy()
    
    for _ in range(7):
        pred = model.predict(last_row)[0]
        future_prices.append(pred)
        # Update last_row for next step (Simulated)
        last_row['Close'] = pred
        # Note: In real production, we'd need to re-calc indicators. This is a fast approximation.
    
    return future_prices

# ==========================================
# 4. REPORT ENGINE
# ==========================================
def create_report(fund, df, ml_preds):
    html = f"""
    <html><body style="font-family:sans-serif; padding:30px; color:#333;">
        <h1 style="border-bottom:2px solid #333;">{fund['symbol']} Institutional Report</h1>
        <div style="background:{'#d4edda' if 'BUY' in fund['verdict'] else '#f8d7da'}; padding:15px; border-radius:5px; margin:20px 0;">
            <h2>AI VERDICT: {fund['verdict']} ({fund['win_prob']:.0f}% Probability)</h2>
        </div>
        
        <h3>📊 Intraday Levels (Day Trading)</h3>
        <table style="width:100%; text-align:left; border-collapse:collapse;">
            <tr style="background:#eee;"><th>Level</th><th>Price</th><th>Action</th></tr>
            <tr><td>Resistance (R1)</td><td>{fund['pivot']['R1']:.2f}</td><td>Sell Zone (Short)</td></tr>
            <tr><td>Pivot Point</td><td>{fund['pivot']['P']:.2f}</td><td>Neutral</td></tr>
            <tr><td>Support (S1)</td><td>{fund['pivot']['S1']:.2f}</td><td>Buy Zone (Long)</td></tr>
        </table>
        
        <h3>🤖 ML Forecast (Next 7 Days)</h3>
        <p>The AI Model (Random Forest) predicts a move to: <b>{fund['currency']} {ml_preds[-1]:.2f}</b></p>
        
        <h3>📉 Short Selling Opportunity?</h3>
        <p>{'YES. Asset is Overbought + Bearish Trend.' if fund['signals']['Trend'] == 'Down' and fund['signals']['Bollinger'] == 'Overbought' else 'NO. Trend is currently strong.'}</p>
        
        <p style="margin-top:50px; color:#888; font-size:12px;">Generated by StockOracle Ultra.</p>
    </body></html>
    """
    return html

# ==========================================
# 5. UI DASHBOARD
# ==========================================
with st.sidebar:
    st.title("🎛 Control Room")
    q_sel = st.selectbox("Quick Load:", ["Select...", "RELIANCE", "TATA MOTORS", "ZOMATO", "APPLE", "BITCOIN"])
    if q_sel != "Select...": st.session_state.sel = q_sel

st.title("🚀 StockOracle: Ultra Edition")
st.caption("Machine Learning • Intraday Signals • Short Selling Analysis")

query = st.text_input("Search Asset:", value=st.session_state.get('sel', 'RELIANCE'))

if st.button("🚀 Run AI Analysis", type="primary"):
    with st.spinner("Training AI Models & Calculating Pivot Points..."):
        df, data, err = get_complete_data(query)
        
        if err: st.error(err)
        else:
            # --- HEADER ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{data['currency']} {data['price']:.2f}", f"{data['change']:.2f}%")
            c2.metric("Win Probability", f"{data['win_prob']:.0f}%", delta="High Confidence" if data['win_prob']>70 else "Low Confidence")
            c3.metric("ATR (Volatility)", f"{data['volatility']:.2f}")
            c4.metric("Recommendation", data['verdict'])
            
            st.progress(int(data['win_prob']))

            # --- INTRADAY PANEL (New Feature) ---
            st.markdown("### ⚡ Intraday & Short Selling")
            i1, i2, i3 = st.columns(3)
            
            with i1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("**📌 Pivot Points (Today)**")
                st.write(f"Resist (Sell): **{data['pivot']['R1']:.2f}**")
                st.write(f"Support (Buy): **{data['pivot']['S1']:.2f}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with i2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("**📉 Short Sell Signal?**")
                if data['signals']['Trend'] == 'Down' and data['price'] < data['pivot']['P']:
                    st.error("YES. Price below Pivot & Trend Down.")
                else:
                    st.success("NO. Buying pressure is present.")
                st.write(f"VWAP: **{df['VWAP'].iloc[-1]:.2f}**")
                st.markdown('</div>', unsafe_allow_html=True)

            with i3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("**🤖 ML Prediction (7 Days)**")
                preds = predict_ml(df)
                gain = ((preds[-1] - data['price']) / data['price']) * 100
                color = "green" if gain > 0 else "red"
                st.markdown(f"Target: **{preds[-1]:.2f}**")
                st.markdown(f"Potential: :{color}[{gain:.2f}%]")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- ADVANCED CHART ---
            fig = go.Figure()
            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Lower Band', fill='tonexty'))
            # ML Forecast
            dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
            fig.add_trace(go.Scatter(x=dates, y=preds, line=dict(color='#ff00ff', width=3), name='AI Forecast'))
            
            fig.update_layout(height=600, template="plotly_dark", title=f"AI Analysis: {data['symbol']}")
            st.plotly_chart(fig, use_container_width=True)

            # --- REPORT DOWNLOAD ---
            html = create_report(data, df, preds)
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{data["symbol"]}_Ultra_Report.html" style="text-decoration:none; color:black; background:#00CC96; padding:15px 30px; border-radius:8px; font-weight:bold; display:block; text-align:center;">📥 Download Hedge Fund Report</a>'
            st.markdown(href, unsafe_allow_html=True)
