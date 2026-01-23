import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests
from fpdf import FPDF
from nsepython import equity_history, nse_quote
from pycoingecko import CoinGeckoAPI

# ==========================================
# 1. CONFIG & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle AI", page_icon="🔮")

st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #262730; border-radius: 10px; padding: 20px; text-align: center; }
    .ad-banner { background-color: #262730; padding: 15px; text-align: center; color: #FFD700; border: 1px dashed #FFD700; margin-bottom: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Initialize Crypto API
cg = CoinGeckoAPI()

# ==========================================
# 2. PDF ENGINE (Same as before)
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'StockOracle AI Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(ticker, df, name, verdict, reasons):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Analysis for: {name} ({ticker})", 0, 1, 'L')
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'L')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"AI Verdict: {verdict}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Current Price: {df['Close'].iloc[-1]:,.2f}", 0, 1)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Analysis:", 0, 1)
    pdf.set_font("Arial", "", 11)
    for r in reasons:
        pdf.cell(0, 8, f"- {r}", 0, 1)
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 3. HYBRID DATA ENGINE (The Solution)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_hybrid(symbol):
    symbol = symbol.upper().strip()
    
    # --- MODE 1: CRYPTO (CoinGecko) ---
    # Trigger: If symbol looks like "bitcoin" or "ethereum"
    if symbol.lower() in ['bitcoin', 'ethereum', 'dogecoin', 'solana']:
        try:
            # Fetch 365 days of data
            data = cg.get_coin_market_chart_by_id(id=symbol.lower(), vs_currency='usd', days=365)
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df.set_index('Date', inplace=True)
            
            fund = {"name": symbol.upper(), "price": df['Close'].iloc[-1], "source": "CoinGecko"}
            return df, fund
        except Exception as e:
            st.error(f"Crypto Error: {e}")
            return None, None

    # --- MODE 2: INDIA (NSEPython) ---
    # Trigger: If symbol doesn't have a suffix or ends in .NS
    if ".NS" in symbol or (symbol.isalpha() and len(symbol) < 10): 
        clean_sym = symbol.replace(".NS", "")
        try:
            # 1. Fetch History
            # Note: NSEPython history can be tricky on cloud, we use a robust range
            series = equity_history(clean_sym, "EQ", "01-01-2024", datetime.now().strftime("%d-%m-%Y"))
            
            # If NSEPython fails (common on cloud IPs), we fallback to Yahoo logic below
            if series is None or len(series) < 5:
                raise ValueError("NSE Blocked")

            df = pd.DataFrame(series)
            # NSEPython returns different columns, need to map them
            df = df.rename(columns={'CH_TIMESTAMP': 'Date', 'CH_CLOSING_PRICE': 'Close'})
            df['Date'] = pd.to_datetime(df['Date'])
            df['Close'] = df['Close'].astype(float)
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            
            fund = {"name": clean_sym, "price": df['Close'].iloc[-1], "source": "NSE India"}
            return df, fund
        except:
            pass # Fallback to Yahoo if NSE fails

    # --- MODE 3: GLOBAL/FALLBACK (Yahoo with Session) ---
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"})
        
        # Try to robustly handle .NS for Yahoo
        yf_sym = symbol if ".NS" in symbol or "-" in symbol else f"{symbol}.NS"
        
        stock = yf.Ticker(yf_sym, session=session)
        hist = stock.history(period="1y")
        
        if hist.empty:
            # Try without .NS (e.g., US stocks)
            stock = yf.Ticker(symbol, session=session)
            hist = stock.history(period="1y")
            
        if hist.empty: return None, None
        
        info = stock.info
        name = info.get('longName', symbol)
        fund = {"name": name, "price": hist['Close'].iloc[-1], "source": "Yahoo"}
        return hist, fund
    except Exception as e:
        return None, None

# ==========================================
# 4. ANALYSIS LOGIC
# ==========================================
def analyze_stock(df):
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    last = df.iloc[-1]
    score = 0
    reasons = []
    
    if last['Close'] > (last['EMA_200'] if not pd.isna(last['EMA_200']) else 0): 
        score += 2; reasons.append("Price > 200 EMA (Uptrend)")
    else: 
        score -= 2; reasons.append("Price < 200 EMA (Downtrend)")
        
    if last['RSI'] < 30: score += 1; reasons.append("RSI Oversold (Buy)")
    elif last['RSI'] > 70: score -= 1; reasons.append("RSI Overbought (Sell)")
    
    if score >= 2: verdict = "BUY"
    elif score <= -2: verdict = "SELL"
    else: verdict = "HOLD"
    return verdict, reasons

def run_prediction(df):
    try:
        df = df.reset_index()
        df['Ord'] = df['Date'].apply(lambda x: x.toordinal())
        model = LinearRegression().fit(df[['Ord']].values, df['Close'].values)
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
        future_pred = model.predict(np.array([d.toordinal() for d in future_dates]).reshape(-1, 1))
        return future_dates, future_pred
    except: return [], []

# ==========================================
# 5. UI
# ==========================================
st.sidebar.title("💎 StockOracle Hybrid")
st.sidebar.info("Now using **Hybrid Engine**:\n1. NSE India Direct\n2. CoinGecko Crypto\n3. Yahoo Backup")
st.sidebar.markdown("---")
st.sidebar.markdown("[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com)")

st.title("🔮 StockOracle: Hybrid AI Analyst")
st.markdown('<div class="ad-banner">📢 <b>Zero-Brokerage Demat Account</b>: Sign up with [Your Link] for free Analysis!</div>', unsafe_allow_html=True)

# Input
symbol_input = st.text_input("Enter Symbol (e.g., RELIANCE, bitcoin, AAPL):", "RELIANCE")

if st.button("🚀 Analyze"):
    with st.spinner(f"Routing request for {symbol_input}..."):
        df, fund = get_data_hybrid(symbol_input)
        
        if df is not None and not df.empty:
            verdict, reasons = analyze_stock(df)
            f_date, f_price = run_prediction(df)
            
            # Dashboard
            st.success(f"✅ Data fetched from: **{fund['source']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Asset", fund['name'])
            c2.metric("Price", f"{fund['price']:,.2f}")
            color = "green" if "BUY" in verdict else "red" if "SELL" in verdict else "gray"
            c3.markdown(f"## :{color}[{verdict}]")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='History', line=dict(color='blue')))
            if len(f_date) > 0:
                fig.add_trace(go.Scatter(x=f_date, y=f_price, name='AI Forecast', line=dict(color='orange', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Report
            st.subheader("📝 Analysis Report")
            c_gen, c_view = st.columns([1, 2])
            with c_gen:
                pdf_bytes = create_pdf_report(symbol_input, df, fund['name'], verdict, reasons)
                st.download_button("📄 Download PDF", pdf_bytes, f"{symbol_input}_Report.pdf", "application/pdf")
            with c_view:
                for r in reasons: st.write(f"• {r}")
        else:
            st.error("❌ All data sources failed. The cloud IP is heavily blocked.")
            st.info("Try: 'bitcoin' or 'AAPL' to test different sources.")