import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from fpdf import FPDF
from textblob import TextBlob
import time
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import tempfile
import yfinance as yf

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Ultimate", page_icon="📈")

# 👇👇 PASTE YOUR TWELVE DATA API KEY HERE 👇👇
TWELVE_DATA_KEY = "1e345639f9b44da9bd71ffc51b63c9ee" 

st.markdown("""
<style>
    /* Dark Theme & Glass UI */
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00CC96; }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Ad Banners */
    .ad-banner {
        background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
        border: 1px dashed #FFD700;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
        cursor: pointer;
    }
    
    /* Sidebar */
    .watchlist-btn { border: 1px solid #333; margin: 2px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE SETUP
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_tier' not in st.session_state: st.session_state.user_tier = 'Guest'
if 'favorites' not in st.session_state: st.session_state.favorites = ["RELIANCE", "TCS", "BTC-USD"]

# ==========================================
# 3. HYBRID DATA ENGINE (The Core Logic)
# ==========================================
@st.cache_data(ttl=900)
def get_data(symbol):
    # Detect Region & Currency
    is_india = ".NS" in symbol or ".BO" in symbol
    currency = "₹" if is_india else "$"
    
    # Twelve Data Clean Symbol (RELIANCE.NS -> RELIANCE)
    td_symbol = symbol.upper().replace(".NS", "").strip()
    
    df = None
    fund = {}
    source_used = ""

    # --- ATTEMPT 1: TWELVE DATA (Fast API) ---
    if TWELVE_DATA_KEY != "PASTE_YOUR_API_KEY_HERE":
        try:
            url = f"https://api.twelvedata.com/time_series?symbol={td_symbol}&interval=1day&outputsize=365&apikey={TWELVE_DATA_KEY}"
            if "BTC" in symbol: 
                url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval=1day&outputsize=365&apikey={TWELVE_DATA_KEY}"
            
            response = requests.get(url).json()
            
            if "values" in response:
                df = pd.DataFrame(response['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
                df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
                df = df.astype(float)
                source_used = "TwelveData (API)"
        except:
            pass # Silently fail to backup

    # --- ATTEMPT 2: YAHOO FINANCE (Backup / Deep Data) ---
    if df is None or df.empty:
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"})
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period="2y")
            
            if not df.empty:
                source_used = "YahooFinance (Backup)"
                # Fix Yahoo Columns if needed
                if 'Date' in df.columns: df = df.set_index('Date')
        except:
            pass

    # --- IF ALL FAIL ---
    if df is None or df.empty:
        return None, "Stock not found. Try adding .NS (e.g., TATASTEEL.NS)", None

    # --- ANALYSIS LOGIC ---
    try:
        # 1. Technical Indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # New Stock Handling (Avoids crash for recent IPOs)
        if len(df) > 200:
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['EMA_200'] = ta.ema(df['Close'], length=200)
        else:
            df['EMA_50'] = df['Close']
            df['EMA_200'] = df['Close']
            
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)

        # 2. Verdict & Signals
        last = df.iloc[-1]
        score = 0
        signals = []

        # Trend
        if len(df) > 200:
            if last['Close'] > last['EMA_200']: score += 2; signals.append("Trend: Bullish (>200 EMA)")
            else: score -= 2; signals.append("Trend: Bearish (<200 EMA)")
        else:
            signals.append("⚠️ New IPO: Long-term trend data unavailable")

        # Momentum
        if last['RSI'] < 30: score += 1; signals.append("RSI: Oversold (Buy Dip)")
        elif last['RSI'] > 70: score -= 1; signals.append("RSI: Overbought (Caution)")

        # MACD
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            score += 1; signals.append("MACD: Bullish Crossover")
        
        # Final Verdict
        if score >= 2: verdict = "BUY 🚀"
        elif score <= -2: verdict = "SELL 🔻"
        else: verdict = "HOLD ✋"

        # Volatility
        daily_ret = df['Close'].pct_change()
        vol = daily_ret.std() * np.sqrt(252) * 100
        vol_str = f"{vol:.1f}% ({'High' if vol > 30 else 'Stable'})"

        fund = {
            "name": symbol.upper(),
            "price": last['Close'],
            "change": (last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100,
            "verdict": verdict,
            "signals": signals,
            "volatility": vol_str,
            "source": source_used
        }
        return df, fund, currency

    except Exception as e:
        return None, f"Analysis Error: {str(e)}", None

# ==========================================
# 4. PREDICTION ENGINE (60 Days)
# ==========================================
def predict_future(df):
    try:
        df = df.reset_index()
        # Find the date column (can vary between sources)
        date_col = 'Date' if 'Date' in df.columns else 'datetime'
        if date_col not in df.columns: date_col = df.columns[0] # Fallback to first col
        
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
    except:
        return [], []

# ==========================================
# 5. PDF REPORT ENGINE (Crash Proof)
# ==========================================
class ProPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'StockOracle AI | Institutional Research', 0, 1, 'R')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 0, 'C')

def create_pro_report(ticker, fund, df, fig, prediction_data, currency):
    pdf = ProPDF()
    pdf.add_page()
    
    # 1. Header Info
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, f"{fund['name']} ({ticker})", 0, 1)
    
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d %b %Y')} | Currency: {currency}", 0, 1)
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    # 2. Verdict
    pdf.set_font("Arial", "B", 16)
    color = (0, 150, 0) if "BUY" in fund['verdict'] else (200, 0, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"AI VERDICT: {fund['verdict']}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    # 3. Metrics Table
    pdf.set_font("Arial", "", 11)
    target = prediction_data[1][-1] if len(prediction_data[1]) > 0 else 0
    metrics = [
        ["Current Price", f"{currency} {fund['price']:.2f}"],
        ["Target (60 Days)", f"{currency} {target:.2f}"],
        ["Volatility", fund['volatility']],
        ["Data Source", fund['source']]
    ]
    for row in metrics:
        pdf.cell(95, 10, row[0], 1, 0)
        pdf.cell(95, 10, row[1], 1, 1)
    pdf.ln(10)

    # 4. Chart (With Failsafe)
    try:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Technical Chart", 0, 1)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name, scale=1.5, engine="kaleido") 
            pdf.image(tmpfile.name, x=10, w=190)
    except Exception as e:
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 10, f"[Chart skipped: Server constraints. View live on web]", 0, 1)
        pdf.set_text_color(0, 0, 0)
        
    pdf.ln(5)
    
    # 5. Signals
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Analysis Signals", 0, 1)
    pdf.set_font("Arial", "", 11)
    for s in fund['signals']:
        pdf.cell(0, 8, f"- {s}", 0, 1)

    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 6. AUTHENTICATION & UI FLOW
# ==========================================
def login_screen():
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.title("🔐 StockOracle Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                if u.lower() == "admin" and p == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Pro"
                    st.rerun()
                elif u.lower() == "user" and p == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Free"
                    st.rerun()
                else: st.error("Try: admin/123")
        
        st.markdown("---")
        if st.button("🚀 Continue as Guest (Skip Login)"):
            st.session_state.logged_in = True
            st.session_state.user_tier = "Guest"
            st.rerun()

# ==========================================
# 7. MAIN UI
# ==========================================
if not st.session_state.logged_in:
    login_screen()
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("📊 Control Panel")
        st.caption(f"Status: {st.session_state.user_tier}")
        
        # Favorites System
        st.subheader("⭐ Watchlist")
        new_fav = st.text_input("Add Symbol", placeholder="e.g. TATASTEEL").upper()
        if st.button("Add"):
            if new_fav and new_fav not in st.session_state.favorites:
                st.session_state.favorites.append(new_fav)
        
        for fav in st.session_state.favorites:
            if st.button(f"🔍 {fav}", key=fav):
                st.session_state.selected_symbol = fav
        
        st.divider()
        if st.session_state.user_tier != "Pro":
            st.info("🔓 Unlock PDF Reports")
            if st.button("💎 Upgrade to Pro"):
                with st.spinner("Processing..."):
                    time.sleep(1)
                    st.session_state.user_tier = "Pro"
                    st.rerun()
        
        if st.button("Log Out"): 
            st.session_state.logged_in = False
            st.rerun()

    # --- MAIN DASHBOARD ---
    # Ad Banner for Free/Guest
    if st.session_state.user_tier != "Pro":
        st.markdown("""
        <a href="https://zerodha.com/open-account" target="_blank" style="text-decoration:none;">
            <div class="ad-banner">
                <div style="color:#FFD700; font-weight:bold; font-size:1.1rem">📢 OPEN FREE DEMAT ACCOUNT</div>
                <div style="color:#aaa; font-size:0.8rem">Zero Brokerage for 30 Days • Click Here</div>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Analysis", "⚖️ Comparison"])

    # --- TAB 1: ANALYSIS ---
    with tab1:
        # Check if coming from sidebar click
        default_sym = st.session_state.get('selected_symbol', 'RELIANCE.NS')
        symbol = st.text_input("Enter Symbol:", value=default_sym).upper()
        
        if st.button("Analyze Stock", type="primary"):
            with st.spinner("Analyzing Market Data..."):
                df, fund, curr = get_data(symbol)
                
                if df is not None:
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"{curr} {fund['price']:.2f}", f"{fund['change']:.2f}%")
                    m2.metric("RSI", f"{df['RSI'].iloc[-1]:.0f}")
                    m3.metric("Volatility", fund['volatility'])
                    m4.metric("Verdict", fund['verdict'])
                    
                    st.caption(f"Data Source: {fund['source']}")

                    # Charts
                    f_dates, f_prices = predict_future(df)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='History'))
                    fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='60-Day AI Forecast', line=dict(color='purple', dash='dot')))
                    
                    if 'EMA_50' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange'), name='50 EMA'))
                    
                    fig.update_layout(height=500, template="plotly_dark", title="Technical Chart")
                    st.plotly_chart(fig, use_container_width=True)

                    # Signals
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.subheader("🤖 AI Signals")
                        for s in fund['signals']: st.write(f"• {s}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.subheader("🔮 Prediction Target")
                        if len(f_prices) > 0:
                            st.metric("Target (60 Days)", f"{curr} {f_prices[-1]:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # PDF Download
                    st.divider()
                    if st.session_state.user_tier == "Pro":
                        pdf_data = create_pro_report(symbol, fund, df, fig, (f_dates, f_prices), curr)
                        st.download_button("📥 Download Pro PDF", pdf_data, f"{symbol}_Report.pdf", "application/pdf")
                    else:
                        st.warning("🔒 PDF Reports are for PRO users.")
                        st.button("Upgrade to Download")

                else:
                    st.error(fund) # Display error message
    
    # --- TAB 2: COMPARISON ---
    with tab2:
        st.subheader("Compare Assets (Up to 5)")
        comp_input = st.text_input("Symbols (comma separated)", "RELIANCE.NS, TCS.NS, BTC-USD")
        
        if st.button("Run Comparison"):
            tickers = [t.strip().upper() for t in comp_input.split(',')]
            comp_df = pd.DataFrame()
            
            progress = st.progress(0)
            for i, t in enumerate(tickers):
                d, _, _ = get_data(t)
                if d is not None:
                    # Normalize start to 0%
                    norm = (d['Close'] / d['Close'].iloc[0] - 1) * 100
                    comp_df[t] = norm
                time.sleep(0.5) # Prevent API rate limits
                progress.progress((i + 1) / len(tickers))
            
            st.line_chart(comp_df)
            st.caption("Y-Axis: Percentage Return (%) over selected period")
