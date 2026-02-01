import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from fpdf import FPDF
from textblob import TextBlob
import time
import plotly.graph_objects as go

# ==========================================
# 1. CONFIG & API SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle SaaS", page_icon="📈")

# 🔑 PASTE YOUR TWELVE DATA API KEY HERE
# Get it for free at: https://twelvedata.com/
API_KEY = "1e345639f9b44da9bd71ffc51b63c9ee" 

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .ad-banner { background-color: #262730; border: 2px dashed #FFD700; padding: 15px; text-align: center; border-radius: 10px; margin-bottom: 20px; transition: transform 0.2s; }
    .ad-banner:hover { transform: scale(1.01); background-color: #333; }
    .lock-box { border: 1px solid #444; background: rgba(255, 255, 255, 0.05); padding: 40px; text-align: center; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIN SYSTEM
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_tier' not in st.session_state: st.session_state.user_tier = 'Free'
if 'username' not in st.session_state: st.session_state.username = ''

def login_screen():
    st.title("🔐 Login to StockOracle")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                if u.lower() == "admin" and p == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Pro"
                    st.session_state.username = "Admin"
                    st.rerun()
                elif u.lower() == "user" and p == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Free"
                    st.session_state.username = "User"
                    st.rerun()
                else: st.error("Try: user/123 or admin/123")
    st.info("💡 **Demo Logins:** `user` / `123` (Free) | `admin` / `123` (Pro)")

def logout():
    st.session_state.logged_in = False
    st.session_state.user_tier = 'Free'
    st.rerun()

# ==========================================
# 3. PROFESSIONAL DATA ENGINE (Twelve Data)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_pro(symbol):
    # Clean symbol: TwelveData uses "RELIANCE" not "RELIANCE.NS" for India
    clean_symbol = symbol.replace(".NS", "").upper()
    
    # URL construction
    url = f"https://api.twelvedata.com/time_series?symbol={clean_symbol}&interval=1day&outputsize=365&apikey={API_KEY}"
    
    # If using Indian stocks, sometimes need to specify exchange, but usually auto-detected
    if "BTC" in clean_symbol: url = f"https://api.twelvedata.com/time_series?symbol=BTC/USD&interval=1day&outputsize=365&apikey={API_KEY}"
    
    try:
        response = requests.get(url).json()
        
        if "values" not in response:
            return None, response.get("message", "Unknown Error")
            
        # Parse Data
        df = pd.DataFrame(response['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        df = df.astype(float)
        
        # Calculate Indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        fund = {
            "name": clean_symbol,
            "price": df['Close'].iloc[-1],
            "pe": "N/A (API Limit)", # Basic tier doesn't give P/E
            "news": "Market data fetched successfully."
        }
        return df, fund
    except Exception as e:
        return None, str(e)

# ==========================================
# 4. REPORT ENGINE
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'StockOracle Pro Report', 0, 1, 'C')
        self.ln(10)

def create_pdf(ticker, fund, hist, tier):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Asset: {ticker} | Price: {fund['price']:.2f}", 0, 1)
    
    if tier == "Pro":
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "PRO: Technical Analysis", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"- RSI (14): {hist['RSI'].iloc[-1]:.2f}", 0, 1)
        
        macd_val = hist['MACD_12_26_9'].iloc[-1]
        signal = "BULLISH" if macd_val > 0 else "BEARISH"
        pdf.cell(0, 10, f"- MACD Signal: {macd_val:.2f} ({signal})", 0, 1)
    else:
        pdf.ln(20)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Upgrade to PRO for Technical Indicators.", 0, 1, 'C')
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 5. MAIN UI
# ==========================================
if not st.session_state.logged_in:
    login_screen()
else:
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.caption(f"Plan: {st.session_state.user_tier}")
        
        if st.session_state.user_tier == "Free":
            if st.button("💎 Upgrade to PRO"):
                with st.spinner("Upgrading..."):
                    time.sleep(1)
                    st.session_state.user_tier = "Pro"
                    st.rerun()
            
            # SIDEBAR AD
            st.markdown("---")
            st.info("💡 **Go Ad-Free?**")
            st.markdown("[**Open Zero-Brokerage Account**](https://zerodha.com/open-account)")
        else:
            if st.button("Downgrade to Free"):
                st.session_state.user_tier = "Free"
                st.rerun()
        if st.button("Logout"): logout()

    # DASHBOARD
    st.title("🔮 StockOracle SaaS")
    
    # TOP AD (Free Only)
    if st.session_state.user_tier == "Free":
        st.markdown("""
        <a href="https://zerodha.com/open-account" target="_blank" style="text-decoration:none;">
            <div class="ad-banner">
                <div style="color:#FFD700; font-weight:bold; font-size:18px;">📢 OPEN DEMAT ACCOUNT: Get ₹0 Brokerage + Free Tools</div>
                <small style="color:#CCC">Sponsored Ad • Click to Support Us</small>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # INPUT
    symbol = st.text_input("Enter Symbol (e.g. RELIANCE, TCS, BTC/USD):", "RELIANCE").upper()

    if st.button("Analyze"):
        if API_KEY == "YOUR_API_KEY_HERE":
            st.error("⚠️ Setup Error: You need to paste your Twelve Data API Key in the code first.")
        else:
            df, fund = get_data_pro(symbol)
            
            if df is not None:
                # 1. Price
                st.subheader(f"{fund['name']} - {fund['price']:.2f}")
                
                # 2. Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00CC96')))
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. PRO Features
                st.divider()
                if st.session_state.user_tier == "Pro":
                    st.subheader("💎 PRO Analysis")
                    c1, c2 = st.columns(2)
                    c1.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                    c2.metric("MACD", f"{df['MACD_12_26_9'].iloc[-1]:.2f}")
                else:
                    st.subheader("💎 PRO Analysis (Locked)")
                    st.markdown("""
                    <div class="lock-box">
                        <h3>🔒 Advanced Analytics Locked</h3>
                        <p>Upgrade to PRO to unlock RSI & MACD Signals.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Report
                st.divider()
                pdf = create_pdf(symbol, fund, df, st.session_state.user_tier)
                st.download_button("📥 Download Report", pdf, "report.pdf", "application/pdf")
                
            else:
                st.error(f"Error fetching data: {fund}")
                st.info("Tip: For India, use just 'RELIANCE' (No .NS). For Crypto, use 'BTC/USD'.")
