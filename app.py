import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import yfinance as yf
from nsepython import equity_history

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Trader", page_icon="📊")

# Watchlist State
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "ZOMATO.NS", "TATAMOTORS.NS"]

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    /* SIGNAL BOXES */
    .signal-card { background: #1f2937; padding: 20px; border-radius: 10px; border-left: 5px solid #555; margin-bottom: 20px; }
    .buy-border { border-left-color: #00CC96 !important; }
    .sell-border { border-left-color: #FF4B4B !important; }
    .wait-border { border-left-color: #FFD700 !important; }
    
    /* TEXT HIGHLIGHTS */
    .big-verdict { font-size: 28px; font-weight: bold; margin-bottom: 5px; }
    .reason-list { font-size: 14px; color: #ccc; line-height: 1.6; }
    .reason-good { color: #00CC96; }
    .reason-bad { color: #FF4B4B; }
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

    # --- INDICATORS & STRATEGY ---
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Pivot Points
    pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
    r1 = (2 * pivot) - prev['Low']
    s1 = (2 * pivot) - prev['High']
    
    # --- LOGIC ENGINE ---
    reasons = []
    score = 0
    
    # 1. Trend Analysis
    if last['Close'] > pivot:
        score += 1
        reasons.append(("✅", "Price is ABOVE daily Pivot Point (Bullish)"))
    else:
        score -= 1
        reasons.append(("❌", "Price is BELOW daily Pivot Point (Bearish)"))
        
    if last['Close'] > df['EMA_200'].iloc[-1]:
        score += 1
        reasons.append(("✅", "Long-term Trend is UP (>200 EMA)"))
    else:
        score -= 1
        reasons.append(("❌", "Long-term Trend is DOWN (<200 EMA)"))
        
    # 2. Momentum
    if last['RSI'] < 30:
        score += 1
        reasons.append(("⚠️", "RSI is Oversold (Potential Bounce)"))
    elif last['RSI'] > 70:
        score -= 1
        reasons.append(("⚠️", "RSI is Overbought (Caution)"))
    else:
        reasons.append(("ℹ️", f"RSI is Neutral ({last['RSI']:.0f})"))

    # 3. Decision
    atr = df['ATR'].iloc[-1]
    if score >= 2:
        verdict = "BUY / LONG"
        style = "buy-border"
        stop_loss = last['Close'] - (1.5 * atr)
        target = last['Close'] + (2.0 * atr)
    elif score <= -2:
        verdict = "SELL / SHORT"
        style = "sell-border"
        stop_loss = last['Close'] + (1.5 * atr)
        target = last['Close'] - (2.0 * atr)
    else:
        verdict = "WAIT / NEUTRAL"
        style = "wait-border"
        stop_loss = last['Close'] * 0.99
        target = last['Close'] * 1.01

    # Risk Reward
    risk = abs(last['Close'] - stop_loss)
    reward = abs(target - last['Close'])
    rr_ratio = reward / risk if risk > 0 else 0

    data = {
        "symbol": symbol,
        "price": last['Close'],
        "currency": "₹" if ".NS" in symbol else "$",
        "change": ((last['Close'] - prev['Close'])/prev['Close']) * 100,
        "verdict": verdict,
        "style": style,
        "reasons": reasons,
        "pivot": pivot,
        "r1": r1, "s1": s1,
        "stop_loss": stop_loss,
        "target": target,
        "rr_ratio": rr_ratio,
        "rsi": last['RSI'],
        "source": source
    }
    return df, data, None

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.title("📊 StockOracle: Trade Assistant")
st.caption("Detailed Analysis • Risk/Reward Calculation • Execution Plan")

tab1, tab2 = st.tabs(["🔍 Deep Dive", "📋 Watchlist"])

# --- TAB 1: EXPLAINABLE ANALYSIS ---
with tab1:
    c_in, c_go = st.columns([3, 1])
    with c_in: q = st.text_input("Analyze Asset:", "HCLTECH")
    with c_go: 
        if st.button("Generate Plan", type="primary"):
            st.session_state.run_analysis = True

    if st.session_state.get('run_analysis', False):
        with st.spinner("Analyzing Market Structure..."):
            df, data, err = get_data(q)
            if err:
                st.error(err)
            else:
                # 1. TOP METRICS
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"{data['currency']} {data['price']:.2f}", f"{data['change']:.2f}%")
                m2.metric("RSI Strength", f"{data['rsi']:.0f}/100")
                m3.metric("Risk/Reward", f"1:{data['rr_ratio']:.1f}")
                m4.metric("Source", data['source'])

                # 2. MAIN STRATEGY CARD
                st.markdown(f"""
                <div class="signal-card {data['style']}">
                    <div class="big-verdict">{data['verdict']}</div>
                    <div style="margin-bottom:10px; color:#aaa;">Confidence Score based on Trend, Momentum, and Pivot Levels.</div>
                    <div class="reason-list">
                        {'<br>'.join([f'<span class="{ "reason-good" if "✅" in r[0] else "reason-bad" if "❌" in r[0] else "" }">{r[0]} {r[1]}</span>' for r in data['reasons']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 3. EXECUTION PLAN (The "Profit" Part)
                c_ex1, c_ex2 = st.columns(2)
                
                with c_ex1:
                    st.subheader("🎯 Trade Setup")
                    st.info(f"""
                    **Entry Zone:** Market Price ({data['price']:.2f})
                    
                    **⛔ Stop Loss:** {data['stop_loss']:.2f}
                    *(Exit if price closes below this)*
                    
                    **💰 Target:** {data['target']:.2f}
                    *(Book profit at this level)*
                    """)
                
                with c_ex2:
                    st.subheader("🔑 Key Levels")
                    st.write(f"Resistance (R1): **{data['r1']:.2f}**")
                    st.write(f"Pivot Point: **{data['pivot']:.2f}**")
                    st.write(f"Support (S1): **{data['s1']:.2f}**")
                    
                    # Visual Gauge for RSI
                    st.write("Momentum Gauge (RSI):")
                    st.progress(int(data['rsi']))
                    st.caption("0 = Oversold (Buy) | 100 = Overbought (Sell)")

                # 4. CHART
                st.subheader("📉 Technical Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                fig.add_hline(y=data['pivot'], line_dash="dash", line_color="yellow", annotation_text="Pivot")
                fig.add_hline(y=data['stop_loss'], line_color="red", annotation_text="Stop Loss")
                fig.add_hline(y=data['target'], line_color="green", annotation_text="Target")
                fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: WATCHLIST (Keep existing functionality) ---
with tab2:
    st.subheader("Market Watch")
    new_s = st.text_input("Add to Watchlist:", key="wl_in")
    if st.button("Add"):
        if new_s: st.session_state.watchlist.append(parse_symbol(new_s))
    
    st.write("Tracking:")
    cols = st.columns(5)
    for i, s in enumerate(st.session_state.watchlist):
        if cols[i % 5].button(f"❌ {s}"):
            st.session_state.watchlist.remove(s)
            st.rerun()
            
    if st.button("Refresh Table"):
        rows = []
        bar = st.progress(0)
        for i, s in enumerate(st.session_state.watchlist):
            _, d, e = get_data(s)
            if d:
                rows.append({
                    "Symbol": s, 
                    "Price": f"{d['price']:.2f}", 
                    "Action": d['verdict'], 
                    "R/R": f"1:{d['rr_ratio']:.1f}"
                })
            bar.progress((i+1)/len(st.session_state.watchlist))
        if rows:
            st.dataframe(pd.DataFrame(rows).style.applymap(lambda v: 'color: #00CC96; font-weight:bold' if 'BUY' in v else 'color: #FF4B4B; font-weight:bold' if 'SELL' in v else '', subset=['Action']), use_container_width=True)
