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
# 1. APP CONFIG & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle SaaS", page_icon="📈")

st.markdown("""
<style>
    /* Clean Dark Mode Look */
    .stApp { background-color: #0e1117; }
    
    /* AD BANNERS */
    .ad-banner { 
        background-color: #262730; 
        border: 2px dashed #FFD700; 
        padding: 15px; 
        text-align: center; 
        border-radius: 10px; 
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .ad-banner:hover { transform: scale(1.01); background-color: #333; }
    .ad-text { color: #FFD700; font-size: 18px; font-weight: bold; text-decoration: none; }
    
    /* LOCKED CONTENT OVERLAY */
    .lock-box { 
        border: 1px solid #444; 
        background: rgba(255, 255, 255, 0.05); 
        padding: 40px; 
        text-align: center; 
        border-radius: 10px; 
    }
    .blur-text { filter: blur(5px); opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIN SYSTEM (Simple & Fast)
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_tier' not in st.session_state: st.session_state.user_tier = 'Free'
if 'username' not in st.session_state: st.session_state.username = ''

def login_screen():
    st.title("🔐 Login to StockOracle")
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        with st.form("login_form"):
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            
            if submitted:
                # --- TEST CREDENTIALS ---
                if user.lower() == "admin" and password == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Pro"
                    st.session_state.username = "Admin"
                    st.rerun()
                elif user.lower() == "user" and password == "123":
                    st.session_state.logged_in = True
                    st.session_state.user_tier = "Free"
                    st.session_state.username = "User"
                    st.rerun()
                else:
                    st.error("Wrong password. Try: user/123 or admin/123")
    
    st.info("💡 **Demo Logins:** \n* Free Plan: `user` / `123` \n* Pro Plan: `admin` / `123`")

def logout():
    st.session_state.logged_in = False
    st.session_state.user_tier = "Free"
    st.rerun()

# ==========================================
# 3. DATA ENGINE (Optimized)
# ==========================================
@st.cache_data(ttl=600) # Cache for 10 mins to speed up reloading
def get_data(symbol):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        hist = ticker.history(period="1y")
        
        if hist.empty: return None, None
        
        # Fast Technicals
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
# 4. REPORT ENGINE
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'StockOracle Analysis', 0, 1, 'C')
        self.ln(10)

def create_pdf(ticker, fund, hist, tier):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Symbol: {ticker} | Price: {fund['price']:.2f}", 0, 1)
    
    if tier == "Pro":
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "PRO: Advanced Metrics", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"- RSI Score: {hist['RSI'].iloc[-1]:.2f}", 0, 1)
        pdf.cell(0, 10, f"- MACD Signal: {hist['MACD_12_26_9'].iloc[-1]:.2f}", 0, 1)
        pdf.cell(0, 10, f"- P/E Ratio: {fund['pe']}", 0, 1)
    else:
        pdf.ln(20)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Upgrade to PRO to see Technical Indicators.", 0, 1, 'C')
        
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 5. ADVERTISING ENGINE
# ==========================================
def render_ad(location="top"):
    if st.session_state.user_tier == "Free":
        link = "https://zerodha.com/open-account" # YOUR AFFILIATE LINK
        
        if location == "top":
            st.markdown(f"""
            <a href="{link}" target="_blank" style="text-decoration:none;">
                <div class="ad-banner">
                    <div class="ad-text">📢 OPEN DEMAT ACCOUNT: Get ₹0 Brokerage + Free Tools</div>
                    <small style="color:#CCC">Sponsored Ad • Click to Support Us</small>
                </div>
            </a>
            """, unsafe_allow_html=True)
            
        elif location == "sidebar":
            st.sidebar.markdown("---")
            st.sidebar.info("💡 **Go Ad-Free?**")
            st.sidebar.image("https://placehold.co/300x250/222/FFD700?text=Best+Trading+App", use_column_width=True)
            st.sidebar.markdown(f"[**Sign Up Now**]({link})")

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================
if not st.session_state.logged_in:
    login_screen()
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.caption(f"Plan: {st.session_state.user_tier}")
        
        if st.session_state.user_tier == "Free":
            if st.button("💎 Upgrade to PRO"):
                with st.spinner("Upgrading..."):
                    time.sleep(1)
                    st.session_state.user_tier = "Pro"
                    st.balloons()
                    st.rerun()
            render_ad("sidebar")
        else:
            if st.button("Downgrade to Free"):
                st.session_state.user_tier = "Free"
                st.rerun()
                
        if st.button("Logout"): logout()

    # --- DASHBOARD ---
    st.title("🔮 StockOracle SaaS")
    render_ad("top")

    symbol = st.text_input("Enter Symbol (e.g., RELIANCE.NS, BTC-USD):", "RELIANCE.NS").upper()

    if st.button("Analyze"):
        hist, fund = get_data(symbol)
        
        if hist is not None:
            # 1. Price & Chart (Public)
            st.subheader(f"{fund['name']} - {fund['price']:.2f}")
            st.line_chart(hist['Close'])
            
            # 2. PRO Features (Gated)
            st.divider()
            
            if st.session_state.user_tier == "Pro":
                st.subheader("💎 PRO Analysis")
                c1, c2, c3 = st.columns(3)
                c1.metric("RSI", f"{hist['RSI'].iloc[-1]:.2f}")
                c2.metric("MACD", f"{hist['MACD_12_26_9'].iloc[-1]:.2f}")
                
                news_text = fund['news'][0]['title'] if fund['news'] else ""
                sentiment = TextBlob(news_text).sentiment.polarity if news_text else 0
                c3.metric("Sentiment", f"{sentiment:.2f}")
            else:
                st.subheader("💎 PRO Analysis (Locked)")
                st.markdown("""
                <div class="lock-box">
                    <h3>🔒 Advanced Analytics Locked</h3>
                    <p>Upgrade to PRO to unlock RSI, MACD, and AI Sentiment.</p>
                </div>
                """, unsafe_allow_html=True)

            # 3. Download
            st.divider()
            pdf_bytes = create_pdf(symbol, fund, hist, st.session_state.user_tier)
            st.download_button("📥 Download Report", pdf_bytes, "report.pdf", "application/pdf")
            
        else:
            st.error("Stock not found.")
