import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from nsepython import equity_history

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Workspace", page_icon="🖥️")

# Initialize Session State for Paper Trading
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = []

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
    .bullish-text { color: #00CC96; font-weight: bold; }
    .bearish-text { color: #FF4B4B; font-weight: bold; }
    .pattern-tag {
        background-color: #333;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 5px;
        border: 1px solid #555;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT DATA ENGINE
# ==========================================
def parse_symbol(query):
    MAP = {
        "RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS",
        "ZOMATO": "ZOMATO.NS", "PAYTM": "PAYTM.NS", "HDFC BANK": "HDFCBANK.NS",
        "INFOSYS": "INFY.NS", "ITC": "ITC.NS", "TCS": "TCS.NS",
        "APPLE": "AAPL", "TESLA": "TSLA", "BITCOIN": "BTC-USD", "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"
    }
    s = query.upper().strip()
    s = MAP.get(s, s)
    if s.endswith(".US"): return s.replace(".US", "")
    if s.endswith(".CR"): return s.replace(".CR", "-USD")
    if not any(x in s for x in [".NS", "-", "="]) and len(s) < 9 and s.isalpha():
        return f"{s}.NS"
    return s

@st.cache_data(ttl=300)
def get_data(query):
    symbol = parse_symbol(query)
    df = None
    source = "Yahoo"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
    except: pass

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

    # --- TECHNICALS ---
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Pivot Points
    last = df.iloc[-1]
    prev = df.iloc[-2]
    pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
    r1 = (2 * pivot) - prev['Low']
    s1 = (2 * pivot) - prev['High']
    
    # --- PATTERN RECOGNITION (Custom Logic) ---
    patterns = []
    
    # 1. Hammer (Bullish Reversal)
    # Body is small, Lower shadow is 2x body, Upper shadow is tiny
    body = abs(last['Close'] - last['Open'])
    lower_shadow = last['Open'] - last['Low'] if last['Close'] > last['Open'] else last['Close'] - last['Low']
    upper_shadow = last['High'] - last['Close'] if last['Close'] > last['Open'] else last['High'] - last['Open']
    
    if lower_shadow > (2 * body) and upper_shadow < (0.5 * body):
        patterns.append("🔨 Hammer (Bullish)")
        
    # 2. Shooting Star (Bearish Reversal)
    # Upper shadow 2x body, Lower shadow tiny
    if upper_shadow > (2 * body) and lower_shadow < (0.5 * body):
        patterns.append("☄️ Shooting Star (Bearish)")
        
    # 3. Bullish Engulfing
    # Today green, yesterday red, today completely covers yesterday
    prev_body = abs(prev['Close'] - prev['Open'])
    if last['Close'] > last['Open'] and prev['Close'] < prev['Open']: # Green after Red
        if last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
            patterns.append("🕯️ Bullish Engulfing")

    # --- SIGNAL GENERATION ---
    score = 0
    reasons = []
    
    # Pivot Check
    if last['Close'] > pivot: 
        score += 1
        reasons.append("Above Pivot")
    else: 
        score -= 1
        reasons.append("Below Pivot")
        
    # Trend Check
    if last['Close'] > df['EMA_200'].iloc[-1]:
        score += 1
        reasons.append("Uptrend")
    
    # Pattern Bonus
    if "Bullish" in str(patterns): score += 2
    if "Bearish" in str(patterns): score -= 2
    
    atr = df['ATR'].iloc[-1]
    
    if score >= 2:
        verdict = "BUY"
        color = "#00CC96"
        sl = last['Close'] - (1.5 * atr)
        tgt = last['Close'] + (2.5 * atr)
    elif score <= -2:
        verdict = "SELL"
        color = "#FF4B4B"
        sl = last['Close'] + (1.5 * atr)
        tgt = last['Close'] - (2.5 * atr)
    else:
        verdict = "WAIT"
        color = "#FFD700"
        sl = last['Close'] * 0.99
        tgt = last['Close'] * 1.01

    data = {
        "symbol": symbol,
        "price": last['Close'],
        "currency": "₹" if ".NS" in symbol else "$",
        "change": ((last['Close'] - prev['Close'])/prev['Close']) * 100,
        "verdict": verdict,
        "verdict_color": color,
        "patterns": patterns,
        "pivot": pivot,
        "r1": r1, "s1": s1,
        "stop_loss": sl,
        "target": tgt,
        "atr": atr,
        "rsi": last['RSI'],
        "source": source
    }
    return df, data, None

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.title("🖥️ StockOracle: Trader's Workspace")

# --- SIDEBAR: PAPER TRADING ---
with st.sidebar:
    st.header("📝 Paper Trading")
    st.caption("Practice without real money")
    
    # Simple Portfolio Display
    if st.session_state.paper_portfolio:
        for trade in st.session_state.paper_portfolio:
            with st.expander(f"{trade['symbol']} ({trade['side']})"):
                st.write(f"Entry: {trade['entry']}")
                st.write(f"Qty: {trade['qty']}")
                if st.button("Close Trade", key=f"close_{trade['symbol']}"):
                    st.session_state.paper_portfolio.remove(trade)
                    st.rerun()
    else:
        st.info("No active simulated trades.")

# --- MAIN INPUT ---
c1, c2 = st.columns([3, 1])
with c1: query = st.text_input("Analyze Asset:", "TATAMOTORS")
with c2: 
    if st.button("Scan Market", type="primary"): st.session_state.scan = True

if st.session_state.get('scan', False):
    with st.spinner("Scanning Charts & Patterns..."):
        df, data, err = get_data(query)
        
        if err:
            st.error(err)
        else:
            # 1. HEADLINES
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{data['currency']} {data['price']:.2f}", f"{data['change']:.2f}%")
            m2.metric("RSI (14)", f"{data['rsi']:.0f}")
            m3.metric("Trend", "BULLISH" if data['price'] > df['EMA_200'].iloc[-1] else "BEARISH")
            m4.markdown(f"<div style='background:{data['verdict_color']}; color:black; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>{data['verdict']}</div>", unsafe_allow_html=True)
            
            # 2. PATTERN & SETUP CARD
            st.markdown("### 🔍 Technical Setup")
            c_setup, c_calc = st.columns([1, 1])
            
            with c_setup:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Candlestick Patterns")
                if data['patterns']:
                    for p in data['patterns']:
                        st.markdown(f"<span class='pattern-tag'>{p}</span>", unsafe_allow_html=True)
                else:
                    st.write("No major reversal patterns detected today.")
                
                st.markdown("---")
                st.write(f"**Stop Loss:** {data['stop_loss']:.2f}")
                st.write(f"**Target:** {data['target']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 3. POSITION SIZING CALCULATOR (New Feature)
            with c_calc:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🧮 Position Size Calculator")
                capital = st.number_input("Capital Available (₹/$)", value=50000, step=1000)
                risk_pct = st.slider("Risk per Trade (%)", 1, 5, 2)
                
                risk_amount = capital * (risk_pct / 100)
                risk_per_share = abs(data['price'] - data['stop_loss'])
                
                if risk_per_share > 0:
                    qty = int(risk_amount / risk_per_share)
                    total_value = qty * data['price']
                    
                    st.write(f"Risk Amount: **{data['currency']} {risk_amount:.0f}**")
                    st.markdown(f"### Buy **{qty}** Shares")
                    st.caption(f"Total Value: {data['currency']} {total_value:.0f}")
                    
                    # PAPER TRADE BUTTON
                    if st.button("Simulate This Trade"):
                        st.session_state.paper_portfolio.append({
                            "symbol": data['symbol'],
                            "side": data['verdict'],
                            "entry": data['price'],
                            "qty": qty
                        })
                        st.success(f"Added {qty} {data['symbol']} to Paper Portfolio!")
                else:
                    st.warning("Stop Loss too close to Price.")
                st.markdown('</div>', unsafe_allow_html=True)

            # 4. CHART
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange', width=1), name='50 EMA'))
            fig.add_hline(y=data['pivot'], line_dash="dash", line_color="white", annotation_text="Pivot")
            fig.add_hline(y=data['stop_loss'], line_color="red", annotation_text="SL")
            fig.add_hline(y=data['target'], line_color="green", annotation_text="TGT")
            
            fig.update_layout(height=500, template="plotly_dark", title=f"{data['symbol']} Trading View", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
