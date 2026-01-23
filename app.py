import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# ==========================================
# 1. WEBSITE CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="StockOracle AI", page_icon="🔮")

# Custom CSS for a "Pro" look
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .green { color: #00ff00; }
    .red { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (FUNDAMENTALS + TECHNICALS)
# ==========================================
def get_stock_data(ticker):
    try:
        # Ticker object
        stock = yf.Ticker(ticker)
        
        # Force a fresh download (bypassing cache)
        hist = stock.history(period="2y", auto_adjust=True)
        
        if hist.empty:
            st.error(f"⚠️ No data found for {ticker}. Try 'RELIANCE.NS' or 'BTC-USD'.")
            return None, None
            
        info = stock.info
        fundamentals = {
            "pe_ratio": info.get('forwardPE', 0),
            "market_cap": info.get('marketCap', 0),
            "sector": info.get('sector', 'Unknown'),
            "beta": info.get('beta', 1),
            "name": info.get('longName', ticker)
        }
        return hist, fundamentals
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def run_prediction_model(df, days=30):
    """
    Simple AI: Uses Linear Regression to predict trend.
    """
    df = df.reset_index()
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    
    # Prepare Data
    X = df[['Date_Ordinal']].values
    y = df['Close'].values
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Next 30 Days
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    future_prices = model.predict(future_ordinals)
    
    return future_dates, future_prices, model.coef_[0] # coef is the slope (Trend)

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================
def analyze_stock(df, fund):
    # Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    
    last = df.iloc[-1]
    score = 0
    reasons = []
    
    # 1. Trend Analysis (Technical)
    if last['Close'] > last['EMA_200']:
        score += 2
        reasons.append("✅ Long-Term Uptrend (Price > 200 EMA)")
    else:
        score -= 2
        reasons.append("❌ Long-Term Downtrend (Price < 200 EMA)")
        
    # 2. Momentum (RSI)
    if last['RSI'] < 30:
        score += 1
        reasons.append("✅ Oversold (Good Entry Point)")
    elif last['RSI'] > 70:
        score -= 1
        reasons.append("⚠️ Overbought (Risk of Drop)")
        
    # 3. Fundamental Analysis (The "Invest" Logic)
    pe = fund['pe_ratio']
    if pe > 0 and pe < 25:
        score += 1
        reasons.append(f"✅ Fair Value (P/E: {pe:.1f})")
    elif pe > 50:
        score -= 1
        reasons.append(f"⚠️ Expensive Valuation (P/E: {pe:.1f})")
        
    # Verdict
    if score >= 3: verdict = "STRONG BUY 🚀"
    elif score >= 1: verdict = "BUY 📈"
    elif score <= -2: verdict = "SELL 🔻"
    else: verdict = "HOLD ✋"
    
    return verdict, score, reasons, df

# ==========================================
# 4. THE WEBSITE UI
# ==========================================
st.title("🔮 StockOracle: AI Market Analyst")
st.markdown("Enter any stock symbol (e.g., **RELIANCE.NS**, **AAPL**, **BTC-USD**) to get a deep scan and future prediction.")

# SEARCH BAR
ticker = st.text_input("Enter Symbol:", value="TATAMOTORS.NS").upper()

if st.button("🚀 Analyze Stock"):
    with st.spinner(f"Connecting to satellites... Analyzing {ticker}..."):
        hist, fund = get_stock_data(ticker)
        
        if hist is not None:
            # RUN ANALYSIS
            verdict, score, reasons, df_analyzed = analyze_stock(hist, fund)
            
            # RUN PREDICTION AI
            f_dates, f_prices, trend_slope = run_prediction_model(hist)
            
            # --- SECTION 1: HEADER & VERDICT ---
            st.divider()
            c1, c2, c3 = st.columns([2, 1, 1])
            
            with c1:
                st.subheader(fund['name'])
                curr_price = df_analyzed['Close'].iloc[-1]
                st.metric("Current Price", f"{curr_price:,.2f}")
            
            with c2:
                color = "green" if "BUY" in verdict else "red" if "SELL" in verdict else "gray"
                st.markdown(f"<h2 style='color:{color}; text-align: center;'>{verdict}</h2>", unsafe_allow_html=True)
                st.caption("AI Verdict")
                
            with c3:
                st.markdown(f"<h2 style='text-align: center;'>{score}/5</h2>", unsafe_allow_html=True)
                st.caption("Safety Score")

            # --- SECTION 2: FUTURE PREDICTION CHART ---
            st.subheader("🤖 AI Future Prediction (Next 30 Days)")
            
            fig = go.Figure()
            
            # Historical Data
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='History', line=dict(color='blue', width=2)))
            
            # Prediction Line
            fig.add_trace(go.Scatter(x=f_dates, y=f_prices, name='AI Forecast', line=dict(color='orange', width=3, dash='dash')))
            
            # Add EMA Lines
            fig.add_trace(go.Scatter(x=hist.index, y=df_analyzed['EMA_50'], name='50 EMA', line=dict(color='green', width=1)))
            fig.add_trace(go.Scatter(x=hist.index, y=df_analyzed['EMA_200'], name='200 EMA', line=dict(color='red', width=1)))

            fig.update_layout(height=500, template="plotly_dark", title=f"{ticker} Price Projection")
            st.plotly_chart(fig, use_container_width=True)
            
            if trend_slope > 0:
                st.success(f"📈 AI Trend Analysis: The model detects a POSITIVE trend slope (+{trend_slope:.2f}). Prices are projected to rise.")
            else:
                st.error(f"📉 AI Trend Analysis: The model detects a NEGATIVE trend slope ({trend_slope:.2f}). Prices are projected to fall.")

            # --- SECTION 3: DEEP ANALYSIS ---
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("📝 Why this verdict?")
                for r in reasons:
                    st.write(r)
                    
            with c_right:
                st.subheader("📊 Fundamental Health")
                st.dataframe(pd.DataFrame([
                    {"Metric": "P/E Ratio", "Value": f"{fund['pe_ratio']:.2f}"},
                    {"Metric": "Market Cap", "Value": f"{fund['market_cap']:,}"},
                    {"Metric": "Sector", "Value": fund['sector']},
                    {"Metric": "Beta (Volatility)", "Value": f"{fund['beta']:.2f}"},
                ]), hide_index=True, use_container_width=True)

        else:
            st.error("Could not find stock. Check symbol (Use .NS for India).")