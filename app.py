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

# ==========================================
# 1. CONFIG & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle India", page_icon="🇮🇳")

st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00CC96; }
    
    /* Card UI */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Ad Banner */
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
# 2. ROBUST DATA ENGINE (Indian Stock Fix)
# ==========================================
@st.cache_data(ttl=300)
def get_stock_data(symbol):
    # 1. Force NSE Extension for India
    # If user types "TATASTEEL", we make it "TATASTEEL.NS"
    # If they type "BTC-USD", we leave it alone.
    if not symbol.endswith(".NS") and not "-" in symbol:
        symbol = f"{symbol}.NS"
    
    # Currency Symbol Logic
    currency = "₹" if ".NS" in symbol else "$"

    try:
        # 2. Stealth Request (Mimics a Browser to avoid blocking)
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
        })
        
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(period="2y")
        
        if df.empty:
            return None, None, f"No data found for {symbol}. Try checking the spelling."
        
        # 3. Technical Analysis
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Handling New Stocks (Less than 200 days)
        if len(df) > 200:
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['EMA_200'] = ta.ema(df['Close'], length=200)
        else:
            df['EMA_50'] = df['Close']
            df['EMA_200'] = df['Close']
            
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)

        # 4. Generate Signals
        last = df.iloc[-1]
        score = 0
        signals = []
        
        # Trend
        if last['Close'] > last['EMA_200']: 
            score += 2; signals.append("Trend: Bullish (>200 EMA)")
        else: 
            score -= 2; signals.append("Trend: Bearish (<200 EMA)")
            
        # Momentum
        if last['RSI'] < 30: score += 1; signals.append("Momentum: Oversold (Buy Dip)")
        elif last['RSI'] > 70: score -= 1; signals.append("Momentum: Overbought (Caution)")
        
        # Verdict
        if score >= 2: verdict = "STRONG BUY"
        elif score <= -2: verdict = "SELL"
        else: verdict = "HOLD"

        # 5. Volatility & Info
        vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
        
        info = ticker.info
        fund = {
            "name": info.get('longName', symbol.upper()),
            "ticker": symbol.upper(),
            "price": last['Close'],
            "currency": currency,
            "change": (last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100,
            "verdict": verdict,
            "signals": signals,
            "volatility": f"{vol:.1f}%"
        }
        
        return df, fund, None

    except Exception as e:
        return None, None, str(e)

# ==========================================
# 3. PREDICTION ENGINE (60 Days)
# ==========================================
def predict_60_days(df):
    df = df.reset_index()
    # Handle Date Column
    date_col = df.columns[0] # Usually 'Date' or 'Datetime'
    
    # Linear Regression on Ordinal Dates
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

# ==========================================
# 4. REPORT ENGINE (Crash-Proof HTML)
# ==========================================
def create_html_report(fund, df, prediction_data):
    # This creates a professional HTML file instead of a fragile PDF
    # It supports ALL symbols (₹) and never crashes.
    
    target_price = prediction_data[1][-1]
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica', sans-serif; padding: 40px; color: #333; }}
            .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
            .title {{ font-size: 32px; font-weight: bold; color: #000; }}
            .date {{ color: #777; font-size: 14px; margin-top: 5px; }}
            .verdict-box {{ 
                background: {'#e6fffa' if 'BUY' in fund['verdict'] else '#fff5f5'}; 
                border-left: 5px solid {'#00cc96' if 'BUY' in fund['verdict'] else '#ff4b4b'}; 
                padding: 20px; margin: 20px 0; 
            }}
            .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; }}
            .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #333; }}
            ul {{ line-height: 1.6; }}
            .footer {{ margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">{fund['name']} ({fund['ticker']})</div>
            <div class="date">Generated on {datetime.now().strftime('%d %b %Y')} | StockOracle AI</div>
        </div>

        <div class="verdict-box">
            <h2 style="margin:0">AI VERDICT: {fund['verdict']}</h2>
        </div>

        <h3>Executive Summary</h3>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Current Price</div>
                <div class="metric-val">{fund['currency']} {fund['price']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">60-Day Target</div>
                <div class="metric-val">{fund['currency']} {target_price:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Volatility</div>
                <div class="metric-val">{fund['volatility']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">RSI Momentum</div>
                <div class="metric-val">{df['RSI'].iloc[-1]:.1f}</div>
            </div>
        </div>

        <h3>Technical Analysis Signals</h3>
        <ul>
            {''.join([f'<li>{s}</li>' for s in fund['signals']])}
        </ul>

        <div class="footer">
            Generated by StockOracle SaaS. Not financial advice.
        </div>
    </body>
    </html>
    """
    return html

# ==========================================
# 5. MAIN UI
# ==========================================
# --- SIDEBAR ---
with st.sidebar:
    st.title("📊 Control Panel")
    
    # Guest Mode Logic
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = "Guest"
    
    st.info(f"User Status: **{st.session_state.user_tier}**")
    
    # Favorites
    st.subheader("⭐ Watchlist")
    fav = st.text_input("Add Symbol", placeholder="e.g. INFOSYS")
    if st.button("Add"):
        st.success(f"Added {fav}")

    st.divider()
    if st.session_state.user_tier == "Guest":
        st.markdown("🔒 **Pro Features Locked**")
        if st.button("💎 Unlock Pro"):
             st.session_state.user_tier = "Pro"
             st.rerun()

# --- MAIN PAGE ---
st.title("📈 StockOracle: India Edition")

# Ad Banner
if st.session_state.user_tier == "Guest":
    st.markdown("""
    <div class="ad-banner">
        <span style="color:#FFD700; font-weight:bold;">📢 OPEN FREE DEMAT ACCOUNT</span><br>
        <span style="color:#ccc; font-size:0.9em;">Zero Brokerage for 30 Days • Sign Up Now</span>
    </div>
    """, unsafe_allow_html=True)

# Search Bar
symbol_input = st.text_input("Enter Symbol (e.g. TATASTEEL, RELIANCE, ZOMATO):", "RELIANCE")

if st.button("🚀 Analyze Stock", type="primary"):
    with st.spinner("Connecting to NSE Server..."):
        df, fund, error = get_stock_data(symbol_input)
        
        if error:
            st.error(error)
        else:
            # 1. METRICS ROW
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{fund['currency']} {fund['price']:.2f}", f"{fund['change']:.2f}%")
            c2.metric("Target (60D)", f"{fund['currency']} {predict_60_days(df)[1][-1]:.2f}")
            c3.metric("Volatility", fund['volatility'])
            c4.metric("Verdict", fund['verdict'])
            
            # 2. INTERACTIVE CHART
            f_dates, f_prices = predict_60_days(df)
            
            fig = go.Figure()
            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='History'))
            # Prediction
            fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='AI Forecast', line=dict(color='#ab47bc', width=2, dash='dot')))
            # EMA
            if 'EMA_50' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='50 EMA', line=dict(color='orange', width=1)))

            fig.update_layout(height=500, template="plotly_dark", title=f"{fund['name']} Trend Analysis", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. SIGNALS & REPORT
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🤖 AI Signals")
                for s in fund['signals']:
                    st.write(f"• {s}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with c_right:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📄 Report Center")
                
                # HTML Report Generation
                report_html = create_html_report(fund, df, (f_dates, f_prices))
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{fund["ticker"]}_Report.html" style="text-decoration:none; color:white; background:#00CC96; padding:10px 20px; border-radius:5px; font-weight:bold;">📥 Download Full Report</a>'
                
                if st.session_state.user_tier == "Pro":
                    st.markdown(href, unsafe_allow_html=True)
                    st.caption("Open the downloaded file and press Ctrl+P to save as PDF.")
                else:
                    st.warning("🔒 Reports are locked for Guests.")
                    st.button("Upgrade to Download", key="upg_btn")
                
                st.markdown('</div>', unsafe_allow_html=True)
