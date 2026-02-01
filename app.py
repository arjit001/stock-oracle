import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import yfinance as yf
from nsepython import equity_history
import base64

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Profit", page_icon="💸")

# Global Watchlist State
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "ZOMATO.NS", "TATAMOTORS.NS", "BTC-USD"]

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .buy-sig { color: #00CC96; font-weight: bold; font-size: 1.2rem; }
    .sell-sig { color: #FF4B4B; font-weight: bold; font-size: 1.2rem; }
    .metric-box { text-align: center; background: #1f2937; padding: 10px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA ENGINE
# ==========================================
def parse_symbol(query):
    # Smart Mapping
    MAP = {
        "RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
        "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", "HDFC BANK": "HDFCBANK.NS",
        "INFOSYS": "INFY.NS", "ITC": "ITC.NS", "TCS": "TCS.NS",
        "APPLE": "AAPL", "TESLA": "TSLA", "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD"
    }
    
    s = query.upper().strip()
    s = MAP.get(s, s) # Map if exists
    
    # Auto-Suffix logic
    if s.endswith(".US"): return s.replace(".US", "")
    if s.endswith(".CR"): return s.replace(".CR", "-USD")
    # If no suffix and looks like Indian stock, add .NS
    if not any(x in s for x in [".NS", "-", "="]) and len(s) < 9 and s.isalpha():
        return f"{s}.NS"
    return s

@st.cache_data(ttl=300)
def get_data(query):
    symbol = parse_symbol(query)
    df = None
    source = "Yahoo"

    # 1. Try Yahoo
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y") # 1 Year ensures enough data for 200 EMA
    except: pass

    # 2. Try NSEPython (Fallback)
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

    if df is None or df.empty:
        return None, None, f"Data not found for {symbol}"

    # --- INTRADAY STRATEGY CALCULATIONS ---
    # 1. Pivot Points (Classic)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
    r1 = (2 * pivot) - prev['Low']
    s1 = (2 * pivot) - prev['High']
    
    # 2. Indicators
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # VWAP (Approximation for daily timeframe)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # 3. Intraday Signals
    atr = df['ATR'].iloc[-1]
    curr_price = last['Close']
    
    signal = "NEUTRAL"
    action = "WAIT"
    stop_loss = 0.0
    target = 0.0
    
    # BUY Logic: Price > Pivot AND Price > VWAP
    if curr_price > pivot and curr_price > df['VWAP'].iloc[-1]:
        signal = "BULLISH"
        action = "BUY / LONG"
        stop_loss = curr_price - (1.5 * atr)
        target = curr_price + (2 * atr)
        
    # SELL Logic: Price < Pivot AND Price < VWAP
    elif curr_price < pivot and curr_price < df['VWAP'].iloc[-1]:
        signal = "BEARISH"
        action = "SELL / SHORT"
        stop_loss = curr_price + (1.5 * atr)
        target = curr_price - (2 * atr)

    # 4. ML Prediction (Simple Random Forest)
    try:
        df_ml = df.copy().dropna()
        X = df_ml[['Open', 'High', 'Low', 'Volume']]
        y = df_ml['Close']
        model = RandomForestRegressor(n_estimators=50).fit(X, y)
        pred_price = model.predict([last[['Open', 'High', 'Low', 'Volume']].values])[0]
    except:
        pred_price = curr_price # Fallback

    data = {
        "symbol": symbol,
        "price": curr_price,
        "prev_close": prev['Close'],
        "currency": "₹" if ".NS" in symbol else "$",
        "change": ((curr_price - prev['Close'])/prev['Close']) * 100,
        "pivot": pivot,
        "r1": r1, "s1": s1,
        "signal": signal,
        "action": action,
        "stop_loss": stop_loss,
        "target": target,
        "pred_price": pred_price,
        "source": source
    }
    
    return df, data, None

# ==========================================
# 3. UI LAYOUT
# ==========================================
with st.sidebar:
    st.title("🎛 Control Panel")
    st.info("Mode: **Intraday Profit**")

st.title("💸 StockOracle: Profit Edition")
st.caption("Intraday Signals • Pivot Points • VWAP Strategy")

tab1, tab2 = st.tabs(["🚀 Intraday Analysis", "📊 Comparison Table"])

# --- TAB 1: SINGLE STOCK ---
with tab1:
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        query = st.text_input("Enter Asset:", "ZOMATO")
    with col_btn:
        run = st.button("Analyze Now", type="primary")

    if run:
        with st.spinner("Calculating Intraday Levels..."):
            df, data, err = get_data(query)
            if err:
                st.error(err)
            else:
                # HEADER
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"{data['currency']} {data['price']:.2f}", f"{data['change']:.2f}%")
                m2.metric("Signal", data['signal'], delta=data['action'])
                m3.metric("Stop Loss", f"{data['stop_loss']:.2f}")
                m4.metric("Target", f"{data['target']:.2f}")

                # ACTION CARD
                st.markdown("---")
                c_act, c_lvl = st.columns([1, 1])
                
                with c_act:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("⚡ Recommended Action")
                    if "BUY" in data['action']:
                        st.markdown(f"<div class='buy-sig'>🚀 {data['action']}</div>", unsafe_allow_html=True)
                        st.write("Reason: Price is above Daily Pivot & VWAP.")
                    elif "SELL" in data['action']:
                        st.markdown(f"<div class='sell-sig'>🔻 {data['action']}</div>", unsafe_allow_html=True)
                        st.write("Reason: Price is below Daily Pivot & VWAP.")
                    else:
                        st.write("✋ WAIT. Market is chopping around Pivot.")
                    
                    st.write(f"**AI Predicted Close:** {data['pred_price']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c_lvl:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🎯 Key Levels (Today)")
                    st.write(f"Resistance (R1): **{data['r1']:.2f}**")
                    st.write(f"Pivot Point: **{data['pivot']:.2f}**")
                    st.write(f"Support (S1): **{data['s1']:.2f}**")
                    st.markdown('</div>', unsafe_allow_html=True)

                # CHART
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                # Add Pivot Line
                fig.add_hline(y=data['pivot'], line_dash="dash", line_color="yellow", annotation_text="Pivot")
                fig.update_layout(height=500, template="plotly_dark", title=f"{data['symbol']} Intraday Chart")
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: COMPARISON TABLE (FIXED) ---
with tab2:
    st.subheader("📈 Live Market Watch")
    
    # Input for adding
    new_s = st.text_input("Add Stock to Watchlist:", placeholder="e.g. TATAMOTORS")
    if st.button("Add"):
        if new_s:
            clean = parse_symbol(new_s)
            if clean not in st.session_state.watchlist:
                st.session_state.watchlist.append(clean)
    
    # Remove tags
    st.write("Watchlist:")
    cols = st.columns(6)
    for i, s in enumerate(st.session_state.watchlist):
        if cols[i % 6].button(f"❌ {s}"):
            st.session_state.watchlist.remove(s)
            st.rerun()

    if st.button("🔄 Refresh Table"):
        rows = []
        progress = st.progress(0)
        for i, s in enumerate(st.session_state.watchlist):
            _, d, e = get_data(s)
            if d:
                rows.append({
                    "Symbol": s,
                    "Price": f"{d['currency']} {d['price']:.2f}",
                    "Action": d['action'],
                    "Pivot": f"{d['pivot']:.2f}",
                    "Target": f"{d['target']:.2f}",
                    "AI Pred": f"{d['pred_price']:.2f}"
                })
            progress.progress((i+1)/len(st.session_state.watchlist))
        
        if rows:
            df_res = pd.DataFrame(rows)
            def color_action(val):
                color = '#00CC96' if 'BUY' in val else '#FF4B4B' if 'SELL' in val else 'white'
                return f'color: {color}; font-weight: bold'
            st.dataframe(df_res.style.map(color_action, subset=['Action']), use_container_width=True)
