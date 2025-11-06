import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Paste your STOCKS, GLOSSARY, and TUTORIALS dictionaries here

STOCKS = {
    "amazon": {"symbol": "AMZN", "sector": "Technology", "market_cap": 1800},
    "apple": {"symbol": "AAPL", "sector": "Technology", "market_cap": 3000},
    "google": {"symbol": "GOOGL", "sector": "Technology", "market_cap": 1900},
    "meta": {"symbol": "META", "sector": "Technology", "market_cap": 900},
    "microsoft": {"symbol": "MSFT", "sector": "Technology", "market_cap": 2800},
    "netflix": {"symbol": "NFLX", "sector": "Entertainment", "market_cap": 200},
    "nvidia": {"symbol": "NVDA", "sector": "Technology", "market_cap": 1200}
}

GLOSSARY = {
    "Bull Market": "A market condition where prices are rising or expected to rise.",
    "Bear Market": "A market condition where prices are falling by 20% or more.",
    "Market Cap": "Market Capitalization - total value of a company's outstanding shares.",
    "P/E Ratio": "Price-to-Earnings ratio - measures stock price relative to earnings per share.",
    "Volatility": "The degree of variation in stock prices over time.",
    "Dividend": "A portion of company profits distributed to shareholders.",
    "IPO": "Initial Public Offering - when a company first sells shares to the public.",
    "Blue Chip": "Stock of a large, well-established, financially stable company.",
    "Candlestick": "A chart showing open, high, low, and close prices for a period.",
    "Moving Average": "Average stock price over a specific time period to smooth out fluctuations.",
    "RSI": "Relative Strength Index - momentum indicator measuring speed and change of price movements.",
    "MACD": "Moving Average Convergence Divergence - trend-following momentum indicator.",
    "Support Level": "Price level where stock tends to stop falling and bounce back.",
    "Resistance Level": "Price level where stock tends to stop rising.",
    "Day Trading": "Buying and selling stocks within the same trading day.",
    "Long Position": "Buying stock with expectation that price will increase.",
    "Short Position": "Selling borrowed stock expecting to buy it back at lower price.",
    "ETF": "Exchange-Traded Fund - investment fund traded on stock exchanges.",
    "Index": "Measure of stock market performance (e.g., S&P 500, NASDAQ).",
    "Correlation": "Statistical measure of how two stocks move in relation to each other."
}

TUTORIALS = [
    {
        "title": "Getting Started with Stock Investing",
        "content": "Start by understanding your financial goals and risk tolerance. Open a brokerage account, research companies, and start with small investments. Diversify your portfolio across different sectors.",
        "level": "Beginner"
    },
    {
        "title": "Understanding Stock Charts",
        "content": "Learn to read candlestick charts, volume indicators, and moving averages. Charts help identify trends, support/resistance levels, and potential entry/exit points.",
        "level": "Beginner"
    },
    {
        "title": "Fundamental vs Technical Analysis",
        "content": "Fundamental analysis examines company financials, earnings, and industry position. Technical analysis uses charts and patterns to predict price movements. Use both for informed decisions.",
        "level": "Intermediate"
    },
    {
        "title": "Risk Management Strategies",
        "content": "Never invest more than you can afford to lose. Use stop-loss orders, diversify investments, and maintain a balanced portfolio. The 1% rule: don't risk more than 1% of capital on a single trade.",
        "level": "Intermediate"
    },
    {
        "title": "Advanced Trading Strategies",
        "content": "Learn about options trading, swing trading, and momentum strategies. Understand market indicators, use technical analysis tools, and develop a disciplined trading plan.",
        "level": "Advanced"
    },
    {
        "title": "Building a Long-Term Portfolio",
        "content": "Focus on quality companies with strong fundamentals. Use dollar-cost averaging, reinvest dividends, and think long-term. Rebalance portfolio annually to maintain target allocations.",
        "level": "Beginner"
    }
]

@st.cache_data
def fetch_live_stock_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y", interval="1d")
    if df.empty:
        return None
    df.reset_index(inplace=True)
    return df

@st.cache_data
def get_live_price(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d")
    if hist.empty:
        return None
    return hist['Close'].iloc[-1]

def get_prediction_models(df):
    df = df.copy()
    df['Day'] = np.arange(len(df))
    X = df[['Day']].values
    Y = df['Close'].values

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }

    for model in models.values():
        model.fit(X, Y)

    return models, X, Y

def generate_future_predictions(models, last_day, days_ahead=[1, 2, 7, 30]):
    future_days = np.array([last_day + i for i in days_ahead]).reshape(-1, 1)
    future_preds = {}
    for name, model in models.items():
        future_preds[name] = model.predict(future_days)
    return future_preds, days_ahead

def generate_stock_plots(df, stock_name):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Daily Return'] = df['Close'].pct_change()
    df['Moving Average'] = df['Close'].rolling(window=10).mean()

    models, X_train, Y_train = get_prediction_models(df)
    last_day = X_train[-1, 0]
    future_preds, days_ahead = generate_future_predictions(models, last_day)

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=int(d)) for d in days_ahead]

    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig_price.update_layout(title=f'{stock_name} Close Price', xaxis_title='Date', yaxis_title='Price (USD)', height=500)

    # Candlestick chart
    fig_candle = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                               low=df['Low'], close=df['Close'])])
    fig_candle.update_layout(title=f'{stock_name} Candlestick Chart', height=500)

    # Volume chart
    fig_volume = go.Figure([go.Bar(x=df['Date'], y=df['Volume'], marker_color='orange')])
    fig_volume.update_layout(title=f'{stock_name} Volume', height=400)

    # Moving average and daily return chart
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Moving Average'], mode='lines', name='Moving Average'))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Daily Return'], mode='lines', name='Daily Return'))
    fig_ma.update_layout(title=f'{stock_name} Moving Average & Daily Return', height=500)

    # Prediction plots 
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=Y_train, mode='markers', name='Actual Price'))
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=models['Linear Regression'].predict(X_train), mode='lines', name='Linear Regression'))
    fig_pred.add_trace(go.Scatter(x=df['Date'], y=models['Random Forest'].predict(X_train), mode='lines', name='Random Forest'))
    # future predictions
    fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds['Linear Regression'], mode='markers+text',
                                  marker=dict(symbol='diamond', size=10), text=[f"{x:.2f}" for x in future_preds['Linear Regression']], textposition='top center', name='Future Linear Regression'))
    fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds['Random Forest'], mode='markers+text',
                                  marker=dict(symbol='star', size=10), text=[f"{x:.2f}" for x in future_preds['Random Forest']], textposition='top center', name='Future Random Forest'))
    fig_pred.update_layout(title=f'{stock_name} Predictions', height=500)

    return fig_price, fig_candle, fig_volume, fig_ma, fig_pred

def main():
    st.title("Stock Analysis Dashboard")

    stock_name = st.selectbox("Select a stock", options=[name.capitalize() for name in STOCKS.keys()])
    stock_symbol = STOCKS[stock_name.lower()]['symbol']

    st.write(f"### {stock_name} ({stock_symbol})")

    stock_data = fetch_live_stock_data(stock_symbol)
    if stock_data is None or stock_data.empty:
        st.error("Could not fetch stock data. Please try again later.")
        return

    live_price = get_live_price(stock_symbol)
    if live_price is not None:
        st.metric(label="Live Price (USD)", value=f"${live_price:.2f}")
    else:
        st.warning("Live price not available.")

    fig_price, fig_candle, fig_volume, fig_ma, fig_pred = generate_stock_plots(stock_data, stock_name)

    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_candle, use_container_width=True)
    st.plotly_chart(fig_volume, use_container_width=True)
    st.plotly_chart(fig_ma, use_container_width=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    st.header("Investment Tutorials")
    for tutorial in TUTORIALS:
        with st.expander(f"{tutorial['title']} ({tutorial['level']})"):
            st.write(tutorial['content'])

    st.header("Stock Market Glossary")
    for term, definition in GLOSSARY.items():
        st.markdown(f"**{term}:** {definition}")

if __name__ == "__main__":
    main()
