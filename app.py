import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from fpdf import FPDF
from textblob import TextBlob
import time

# ==========================================
# 1. APP CONFIG & STYLE
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle SaaS", page_icon="📈")

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #444; border-radius: 8px; padding: 15px; text-align: center; }
    .ad-banner-top { background-color: #262730; border: 2px dashed #FFD700; padding: 15px; text-align: center; margin-bottom: 20px; border-radius: 10px; cursor: pointer; }
    .ad-banner-top:hover { background-color: #333; }
    .ad-text { color: #FFD700; font-size: 1.1em; font-weight: bold; text-decoration: none; }
    .ad-subtext { color: #AAA; font-size: 0.9em; }
    .lock-overlay { position: relative; text-align: center; padding: 40px; background: rgba(0,0,0,0.8); border-radius: 10px; border: 1px solid #444; margin-top: 20px; }
    .lock-icon { font-size: 40px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIMPLE SESSION STATE AUTH (No Library Needed)
# ==========================================
# Initialize session variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_tier' not in st.session_state:
    st.session_state.user_tier = 'Free'
if 'username' not in st.session_state:
    st.session_state.username = ''

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Log In"):
        # HARDCODED USERS FOR TESTING
        if user == "admin" and password == "123":
            st.session_state.logged_in = True
            st.session_state.user_tier = "Pro"
            st.session_state.username = "Admin"
            st.rerun()
        elif user == "user" and password == "123":
            st.session_state.logged_in = True
            st.session_state.user_tier = "Free"
            st.session_state.username = "User"
            st.rerun()
        else:
            st.error("Invalid credentials. Try: user/123 or admin/123")

def logout():
    st.session_state.logged_in = False
    st.session_state.user_tier = "Free"
    st.rerun()

# ==========================================
# 3. ADVERTISING ENGINE
# ==========================================
def show_ad(position="top"):
    if st.session_state.user_tier == "Free":
        affiliate_link = "https://zerodha.com/open-account" 
        
        if position == "top":
            st.markdown(f"""
            <a href="{affiliate_link}" target="_blank" style="text-decoration:none;">
                <div class="ad-banner-top">
                    <div class="ad-text">📢 SPONSORED: Open a Demat Account & Get ₹0 Brokerage</div>
                    <div class="ad-subtext">Click here to claim your offer + Free Analysis Tools</div>
                </div>
            </a>
            """, unsafe_allow_html=True)
        elif position == "sidebar":
            st.sidebar.markdown("---")
            st.sidebar.info("💡 **Ad-Free Experience?**")
            st.sidebar.markdown(f"[**Click to Sign Up**]({affiliate_link})")

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data(ttl=900)
def fetch_data(symbol):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"})
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        hist = ticker.history(period="1y")
        
        if hist.empty: return None, None
        
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['EMA_200'] = ta.ema(hist['Close'], length=200)
        macd = ta.macd(hist['Close'])
        hist = pd.concat([hist, macd], axis=1)
        
        info = ticker.info
        fund = {
            "name": info.get('longName', symbol),
            "price": hist['Close'].iloc[-1],
            "pe": info.get('forwardPE', 'N/A'),
            "news": ticker.news
        }
        return hist, fund
    except: return None, None

# ==========================================
# 5. REPORT ENGINE
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'StockOracle Analysis', 0, 1, 'C')
        self.ln(5)

def create_report(ticker, fund, hist, tier):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Report for: {fund['name']}", 0, 1)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Current Price: {fund['price']:,.2f}", 0, 1)
    
    if tier == "Pro":
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "PRO: Deep Dive Analysis", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"- RSI Momentum: {hist['RSI'].iloc[-1]:.2f}", 0, 1)
        pdf.cell(0, 10, f"- MACD Signal: {hist['MACD_12_26_9'].iloc[-1]:.2f}", 0, 1)
    else:
        pdf.ln(20)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Upgrade to PRO for Technical Indicators.", 0, 1, 'C')
        
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 6. MAIN APP FLOW
# ==========================================

# IF NOT LOGGED IN -> SHOW LOGIN SCREEN
if not st.session_state.logged_in:
    login()
else:
    # IF LOGGED IN -> SHOW DASHBOARD
    
    # SIDEBAR
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.write(f"Plan: **{st.session_state.user_tier}**")
        
        if st.session_state.user_tier == "Free":
            st.warning("Limits: Ads Enabled")
            if st.button("💎 Upgrade to PRO ($9/mo)"):
                with st.spinner("Processing..."):
                    time.sleep(1)
                    st.session_state.user_tier = "Pro"
                    st.balloons()
                    st.rerun()
            show_ad("sidebar")
        else:
            st.success("✅ Ad-Free")
            if st.button("Downgrade to Free"):
                st.session_state.user_tier = "Free"
                st.rerun()
        
        st.markdown("---")
        if st.button("Logout"):
            logout()

    # MAIN CONTENT
    st.title("🔮 StockOracle SaaS")
    show_ad("top")

    symbol = st.text_input("Enter Symbol (e.g., RELIANCE.NS, AAPL):", "RELIANCE.NS").upper()

    if st.button("Analyze Stock"):
        hist, fund = fetch_data(symbol)
        
        if hist is not None:
            st.subheader(f"{fund['name']} - {fund['price']:,.2f}")
            st.line_chart(hist['Close'])
            
            st.divider()
            
            # GATED CONTENT LOGIC
            if st.session_state.user_tier == "Pro":
                st.subheader("💎 PRO Analysis (Unlocked)")
                c1, c2, c3 = st.columns(3)
                c1.metric("RSI", f"{hist['RSI'].iloc[-1]:.2f}")
                c2.metric("MACD", f"{hist['MACD_12_26_9'].iloc[-1]:.2f}")
                
                news_title = fund['news'][0]['title'] if fund['news'] else ""
                if news_title:
                    blob = TextBlob(news_title)
                    c3.metric("Sentiment", f"{blob.sentiment.polarity:.2f}")
            else:
                st.subheader("💎 PRO Analysis")
                st.markdown("""
                <div class="lock-overlay">
                    <div class="lock-icon">🔒</div>
                    <h3>Advanced Analytics Locked</h3>
                    <p>Upgrade to see RSI, MACD, and AI News Sentiment.</p>
                </div>
                """, unsafe_allow_html=True)
                st.image("https://placehold.co/800x200/333/666?text=Blurry+Technical+Indicators+(Pro+Only)")

            st.divider()
            st.subheader("📥 Report")
            pdf_bytes = create_report(symbol, fund, hist, st.session_state.user_tier)
            st.download_button("Download PDF", pdf_bytes, f"{symbol}_report.pdf", "application/pdf")
            
        else:
            st.error("Stock not found or connection blocked.")
