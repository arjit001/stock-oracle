import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import requests
from fpdf import FPDF
from textblob import TextBlob
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# ==========================================
# 1. APP CONFIG & STYLE
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle SaaS", page_icon="📈")

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #444; border-radius: 8px; padding: 15px; text-align: center; }
    
    /* ADVERTISING BANNERS */
    .ad-banner-top { background-color: #262730; border: 2px dashed #FFD700; padding: 15px; text-align: center; margin-bottom: 20px; border-radius: 10px; cursor: pointer; }
    .ad-banner-top:hover { background-color: #333; }
    .ad-text { color: #FFD700; font-size: 1.1em; font-weight: bold; text-decoration: none; }
    .ad-subtext { color: #AAA; font-size: 0.9em; }

    /* PRO LOCK OVERLAY */
    .lock-overlay { position: relative; text-align: center; padding: 40px; background: rgba(0,0,0,0.8); border-radius: 10px; border: 1px solid #444; margin-top: 20px; }
    .lock-icon { font-size: 40px; margin-bottom: 10px; }
    .upgrade-btn { background-color: #FFD700; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTHENTICATION & SESSION SETUP
# ==========================================
# Simple Hardcoded Config for Demo (In real life, use a database)
config = {
    'credentials': {
        'usernames': {
            'admin': {'name': 'Admin User', 'password': '123', 'email': 'admin@gmail.com', 'tier': 'Pro'},
            'user': {'name': 'Free User', 'password': '123', 'email': 'user@gmail.com', 'tier': 'Free'}
        }
    },
    'cookie': {'expiry_days': 30, 'key': 'random_signature_key', 'name': 'stockoracle_auth'}
}

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# LOGIN WIDGET
name, authentication_status, username = authenticator.login('main')

if authentication_status is False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status is None:
    st.warning('Please enter your username and password (Try: user / 123)')
    st.stop()

# LOAD USER TIER
user_tier = config['credentials']['usernames'][username].get('tier', 'Free')

# ==========================================
# 3. ADVERTISING ENGINE
# ==========================================
def show_ad(position="top"):
    if user_tier == "Free":
        # REPLACE 'YOUR_AFFILIATE_LINK' WITH REAL LINKS (e.g., Zerodha/Binance)
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
            st.sidebar.image("https://placehold.co/300x250/222/FFD700?text=Best+Crypto+App%0ASign+Up+Now", use_column_width=True)
            st.sidebar.markdown(f"[**Click to Sign Up**]({affiliate_link})")

# ==========================================
# 4. DATA ENGINE (STEALTH MODE)
# ==========================================
@st.cache_data(ttl=900)
def fetch_data(symbol):
    session = requests.Session()
    # Fake a Browser to bypass blocks
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"})
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        hist = ticker.history(period="1y")
        
        if hist.empty: return None, None
        
        # Calculate Technicals
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['EMA_200'] = ta.ema(hist['Close'], length=200)
        
        # MACD (Pro Feature)
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
# 5. PDF REPORT ENGINE
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
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.cell(0, 10, f"Current Price: {fund['price']:,.2f}", 0, 1)
    
    if tier == "Pro":
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "PRO: Deep Dive Analysis", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"- RSI Momentum: {hist['RSI'].iloc[-1]:.2f}", 0, 1)
        pdf.cell(0, 10, f"- MACD Signal: {hist['MACD_12_26_9'].iloc[-1]:.2f}", 0, 1)
        pdf.cell(0, 10, f"- PE Ratio: {fund['pe']}", 0, 1)
    else:
        pdf.ln(20)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Upgrade to PRO for Technical Indicators & Sentiment Analysis.", 0, 1, 'C')
        
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 6. MAIN UI DASHBOARD
# ==========================================
# SIDEBAR
with st.sidebar:
    st.title(f"👤 {name}")
    st.write(f"Plan: **{user_tier}**")
    
    if user_tier == "Free":
        st.warning("Limits: Basic Charts, Ads Enabled")
        # UPGRADE BUTTON (Simulated Payment)
        if st.button("💎 Upgrade to PRO ($9/mo)"):
            with st.spinner("Redirecting to Secure Payment..."):
                time.sleep(2)
                st.success("Payment Success! Refreshing...")
                # In real app, you'd update database here
                config['credentials']['usernames'][username]['tier'] = 'Pro'
                st.balloons()
        show_ad("sidebar")
    else:
        st.success("✅ Ad-Free | AI Enabled")
    
    authenticator.logout('Logout', 'main')

# MAIN CONTENT
st.title("🔮 StockOracle SaaS")

# TOP AD BANNER (Free Users Only)
show_ad("top")

symbol = st.text_input("Enter Symbol (e.g., RELIANCE.NS, AAPL):", "RELIANCE.NS").upper()

if st.button("Analyze Stock"):
    hist, fund = fetch_data(symbol)
    
    if hist is not None:
        # 1. PRICE HEADER
        st.subheader(f"{fund['name']} - {fund['price']:,.2f}")
        
        # 2. CHART (Everyone sees this)
        st.line_chart(hist['Close'])
        
        # 3. PRO FEATURES (Gated Content)
        st.divider()
        if user_tier == "Pro":
            st.subheader("💎 PRO Analysis (Unlocked)")
            c1, c2, c3 = st.columns(3)
            c1.metric("RSI", f"{hist['RSI'].iloc[-1]:.2f}")
            c2.metric("MACD", f"{hist['MACD_12_26_9'].iloc[-1]:.2f}")
            
            # Sentiment Analysis
            news_title = fund['news'][0]['title'] if fund['news'] else ""
            if news_title:
                blob = TextBlob(news_title)
                c3.metric("News Sentiment", f"{blob.sentiment.polarity:.2f}")
                st.caption(f"Latest News: {news_title}")
                
        else:
            # LOCKED VIEW FOR FREE USERS
            st.subheader("💎 PRO Analysis")
            st.markdown("""
            <div class="lock-overlay">
                <div class="lock-icon">🔒</div>
                <h3>Advanced Analytics Locked</h3>
                <p>Upgrade to see RSI, MACD, and AI News Sentiment.</p>
                <p style="color:#FF4B4B;">+ Remove All Ads</p>
            </div>
            """, unsafe_allow_html=True)
            # Blur Effect
            st.image("https://placehold.co/800x200/333/666?text=Blurry+Technical+Indicators+(Pro+Only)")

        # 4. DOWNLOAD REPORT
        st.divider()
        st.subheader("📥 Report")
        pdf_bytes = create_report(symbol, fund, hist, user_tier)
        st.download_button("Download PDF", pdf_bytes, f"{symbol}_report.pdf", "application/pdf")
        
    else:
        st.error("Stock not found or connection blocked.")