import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
from nsepython import equity_history
import time

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Algo", page_icon="🤖")

if 'trade_log' not in st.session_state: st.session_state.trade_log = []
if 'broker_status' not in st.session_state: st.session_state.broker_status = False

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    .glass-card { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 15px; }
    .pattern-tag { background-color: #333; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; margin-right: 5px; border: 1px solid #555; }
    .success-msg { color: #00CC96; font-weight: bold; }
    .error-msg { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BROKER BRIDGE (THE ALGO ENGINE)
# ==========================================
def connect_broker(api_key, client_id, pwd):
    # ---------------------------------------------------------
    # REAL BROKER CODE GOES HERE (e.g., Angel One / Zerodha)
    # ---------------------------------------------------------
    # from smartapi import SmartConnect
    # obj = SmartConnect(api_key=api_key)
    # data = obj.generateSession(client_id, pwd)
    # return obj
    # ---------------------------------------------------------
    
    # SIMULATION FOR DEMO
    time.sleep(1) # Fake network delay
    if len(api_key) > 5:
        return True
    return False

def place_broker_order(symbol, qty, side, order_type="MARKET"):
    # ---------------------------------------------------------
    # REAL ORDER PLACEMENT
    # ---------------------------------------------------------
    # orderparams = {
    #     "variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": "3045",
    #     "transactiontype": side, "exchange": "NSE", "ordertype": order_type,
    #     "producttype": "INTRADAY", "duration": "DAY", "price": "0", "quantity": qty
    # }
    # order_id = broker.placeOrder(orderparams)
    # return order_id
    # ---------------------------------------------------------
    
    # SIMULATION
    time.sleep(1)
    return f"ORD_{int(time.time())}"

# ==========================================
# 3. DATA & ANALYSIS ENGINE
# ==========================================
def parse_symbol(query):
    MAP = {"RELIANCE": "RELIANCE.NS", "TATA MOTORS": "TATAMOTORS.NS", "SBI": "SBIN.NS", "ZOMATO": "ZOMATO.NS", "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
    s = query.upper().strip()
    s = MAP.get(s, s)
    if not any(x in s for x in [".NS", "-", "="]) and len(s) < 9 and s.isalpha(): return f"{s}.NS"
    return s

@st.cache_data(ttl=300)
def get_data(query):
    symbol = parse_symbol(query)
    df = None
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
        except: pass

    if df is None or df.empty: return None, None, "Data Not Found"

    # INDICATORS
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
    
    # SIGNAL LOGIC
    score = 0
    if last['Close'] > pivot: score += 1
    if last['Close'] > df['EMA_200'].iloc[-1]: score += 1
    
    atr = df['ATR'].iloc[-1]
    if score >= 2:
        verdict = "BUY"
        color = "#00CC96"
        sl = last['Close'] - (1.5 * atr)
        tgt = last['Close'] + (2.5 * atr)
    elif score <= 0: # Stricter Sell
        verdict = "SELL"
        color = "#FF4B4B"
        sl = last['Close'] + (1.5 * atr)
        tgt = last['Close'] - (2.5 * atr)
    else:
        verdict = "WAIT"
        color = "#FFD700"
        sl, tgt = 0, 0

    data = {
        "symbol": symbol, "price": last['Close'], "verdict": verdict, "color": color,
        "sl": sl, "tgt": tgt, "rsi": last['RSI'], "pivot": pivot
    }
    return df, data, None

# ==========================================
# 4. UI DASHBOARD
# ==========================================
st.title("🤖 StockOracle: Algo Edition")

# --- SIDEBAR: BROKER LOGIN ---
with st.sidebar:
    st.header("🔐 Broker Connection")
    st.caption("Connect AngelOne / Zerodha")
    
    api_key = st.text_input("API Key", type="password")
    client_id = st.text_input("Client ID")
    
    if st.button("Connect Broker"):
        if connect_broker(api_key, client_id, "pwd"):
            st.session_state.broker_status = True
            st.success("Connected to Broker ✅")
        else:
            st.error("Connection Failed ❌")
            
    st.divider()
    st.write("📜 **Algo Trade Log**")
    for log in st.session_state.trade_log:
        st.caption(log)

# --- MAIN ANALYSIS ---
c1, c2 = st.columns([3, 1])
with c1: query = st.text_input("Asset:", "TATAMOTORS")
with c2: 
    if st.button("Scan", type="primary"): st.session_state.scan = True

if st.session_state.get('scan', False):
    df, data, err = get_data(query)
    if err: st.error(err)
    else:
        # TOP METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"{data['price']:.2f}")
        m2.metric("Signal", data['verdict'])
        m3.metric("Stop Loss", f"{data['sl']:.2f}")
        m4.metric("Target", f"{data['tgt']:.2f}")

        # CHART
        fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        fig.add_hline(y=data['pivot'], line_dash="dash", line_color="white", annotation_text="Pivot")
        fig.add_hline(y=data['sl'], line_color="red", annotation_text="SL")
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # --- THE ALGO EXECUTION CARD ---
        st.markdown("### ⚡ Algo Execution Terminal")
        
        ex_col1, ex_col2 = st.columns([1, 1])
        
        with ex_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("1. Position Sizing")
            capital = st.number_input("Capital (₹)", 50000)
            risk = st.slider("Risk %", 1, 5, 2)
            
            # Auto-Calculate Qty
            risk_amt = capital * (risk/100)
            risk_per_share = abs(data['price'] - data['sl'])
            qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            st.info(f"Recommended Qty: **{qty} shares** (Risk: ₹{risk_amt:.0f})")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with ex_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("2. Execute Trade")
            
            if not st.session_state.broker_status:
                st.warning("⚠️ Broker Disconnected. Running in Simulation Mode.")
            else:
                st.success("✅ Broker Connected. Real Orders Enabled.")
            
            # THE BIG BUTTONS
            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"BUY {qty} QTY", type="primary", disabled=data['verdict']!="BUY" or qty==0):
                    order_id = place_broker_order(data['symbol'], qty, "BUY")
                    st.session_state.trade_log.append(f"🟢 BOUGHT {qty} {data['symbol']} @ {data['price']}")
                    st.success(f"Order Placed! ID: {order_id}")
            
            with b2:
                if st.button(f"SELL {qty} QTY", type="secondary", disabled=data['verdict']!="SELL" or qty==0):
                    order_id = place_broker_order(data['symbol'], qty, "SELL")
                    st.session_state.trade_log.append(f"🔴 SOLD {qty} {data['symbol']} @ {data['price']}")
                    st.error(f"Order Placed! ID: {order_id}")
            
            st.caption("Note: Orders are Market Orders (Intraday)")
            st.markdown('</div>', unsafe_allow_html=True)
