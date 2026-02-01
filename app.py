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
import io
import tempfile

# ==========================================
# 1. APP CONFIG & "GLASS" UI
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle Ultimate", page_icon="📈")

# 🔑 API KEY (Twelve Data) - Get free at twelvedata.com
API_KEY = "1e345639f9b44da9bd71ffc51b63c9ee"

# MODERN CSS
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00CC96; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_tier' not in st.session_state: st.session_state.user_tier = 'Guest'

# ==========================================
# 3. PROFESSIONAL PDF ENGINE (THE UPGRADE)
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
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d")} | NOT FINANCIAL ADVICE', 0, 0, 'C')

def create_pro_report(ticker, fund, df, fig):
    pdf = ProPDF()
    pdf.add_page()
    
    # 1. TITLE HEADER
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, f"{fund['name']} ({ticker})", 0, 1)
    
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Sector Report | Date: {datetime.now().strftime('%d %b %Y')}", 0, 1)
    pdf.line(10, 35, 200, 35)
    pdf.ln(10)
    
    # 2. EXECUTIVE SUMMARY TABLE
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    
    # Verdict Color Logic
    verdict_color = (0, 150, 0) if "BUY" in fund['verdict'] else (200, 0, 0)
    
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(*verdict_color)
    pdf.cell(0, 10, f"AI VERDICT: {fund['verdict']}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset
    
    # Metrics Grid
    pdf.set_font("Arial", "", 11)
    pdf.set_fill_color(240, 240, 240)
    
    metrics = [
        ["Current Price", f"INR {fund['price']:.2f}"],
        ["50-Day Moving Avg", f"INR {df['EMA_50'].iloc[-1]:.2f}"],
        ["200-Day Moving Avg", f"INR {df['EMA_200'].iloc[-1]:.2f}"],
        ["RSI (Momentum)", f"{df['RSI'].iloc[-1]:.1f}"],
        ["MACD Signal", "BULLISH" if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] else "BEARISH"]
    ]
    
    for row in metrics:
        pdf.cell(95, 10, row[0], 1, 0, 'L', 1)
        pdf.cell(95, 10, row[1], 1, 1, 'L', 0)
    
    pdf.ln(10)
    
    # 3. EMBED CHART (The Magic Step)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Price Trend & Technicals", 0, 1)
    
    # Save Plotly figure to a temp image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name, scale=2) # scale=2 for high res
        pdf.image(tmpfile.name, x=10, w=190)
    
    pdf.ln(5)
    
    # 4. AI SIGNALS LIST
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Technical Signals", 0, 1)
    pdf.set_font("Arial", "", 11)
    
    for s in fund['signals']:
        # Bullet point symbol
        pdf.cell(5, 8, chr(149), 0, 0)
        pdf.cell(0, 8, s, 0, 1)

    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 4. DATA & CHART ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_data(symbol):
    clean_symbol = symbol.upper().replace(".NS", "").strip()
    url = f"https://api.twelvedata.com/time_series?symbol={clean_symbol}&interval=1day&outputsize=365&apikey={API_KEY}"
    if "BTC" in clean_symbol: url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval=1day&outputsize=365&apikey={API_KEY}"
    
    try:
        response = requests.get(url).json()
        if "values" not in response: return None, "API Error"
            
        df = pd.DataFrame(response['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        df = df.astype(float)
        
        # Technicals
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        # Verdict
        score = 0
        signals = []
        last = df.iloc[-1]
        
        if last['Close'] > last['EMA_200']: score += 2; signals.append("Price > 200 EMA (Long Term Uptrend)")
        else: score -= 2; signals.append("Price < 200 EMA (Long Term Downtrend)")
        
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 1; signals.append("MACD Bullish Crossover")
        
        verdict = "STRONG BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        
        fund = {
            "name": clean_symbol,
            "price": last['Close'],
            "change": (last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100,
            "verdict": verdict,
            "signals": signals
        }
        return df, fund
    except Exception as e: return None, str(e)

# ==========================================
# 5. UI & LOGIC
# ==========================================
if not st.session_state.logged_in:
    c1, c2, c3 = st.columns([1,1.5,1])
    with c2:
        st.title("📈 StockOracle Ultimate")
        if st.button("🚀 Continue as Guest (Skip Login)"):
            st.session_state.logged_in = True
            st.session_state.user_tier = "Free"
            st.rerun()
else:
    # SIDEBAR
    with st.sidebar:
        st.title("📊 Control")
        st.write(f"Tier: **{st.session_state.user_tier}**")
        if st.session_state.user_tier == "Free":
            if st.button("💎 Unlock Pro Reports"):
                with st.spinner("Processing..."):
                    time.sleep(1)
                    st.session_state.user_tier = "Pro"
                    st.rerun()
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # MAIN
    st.title("📈 StockOracle: Market Terminal")
    
    symbol = st.text_input("Search Symbol:", "RELIANCE").upper()
    
    if st.button("Analyze Stock"):
        with st.spinner("Analyzing..."):
            df, fund = get_data(symbol)
            
            if df is not None:
                # 1. METRICS
                m1, m2, m3 = st.columns(3)
                m1.metric(fund['name'], f"{fund['price']:.2f}", f"{fund['change']:.2f}%")
                m2.metric("RSI", f"{df['RSI'].iloc[-1]:.0f}")
                m3.metric("Verdict", fund['verdict'])
                
                # 2. PLOTLY CHART
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange'), name='50 EMA'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue'), name='200 EMA'))
                fig.update_layout(
                    height=500, 
                    template="plotly_white", # White background looks better in PDF
                    title=f"{fund['name']} - Technical Analysis",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. PDF DOWNLOAD
                st.divider()
                st.subheader("📄 Institutional Report")
                
                if st.session_state.user_tier == "Pro":
                    pdf_data = create_pro_report(symbol, fund, df, fig)
                    st.download_button("📥 Download Pro PDF (With Charts)", pdf_data, f"{symbol}_Pro_Report.pdf", "application/pdf")
                else:
                    st.warning("🔒 Charts inside PDF are a PRO feature.")
                    st.button("Upgrade to Unlock")
            else:
                st.error("Data not found. Try 'RELIANCE' or 'BTC/USD'")
