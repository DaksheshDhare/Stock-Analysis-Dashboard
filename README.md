# Stock-Analysis-Dashboard

This project is an interactive stock analysis dashboard built using Streamlit, Plotly, yFinance, and machine learning models. It allows users to explore historical stock data, view live prices, and get future price predictions using multiple regression models in a visually rich and user-friendly interface.

Features
-Select from popular stocks (Amazon, Apple, Google, Meta, Microsoft, Netflix, Nvidia).
-View one-year historical stock prices with interactive Plotly charts:
-Line chart of closing prices
-Candlestick price chart
-Volume bar chart
-Moving average and daily returns
-Machine Learning predictions from Linear Regression, Random Forest, SVM, and AdaBoost models.
-Side-by-side chart layout and large format charts for better visualization.
-Investment tutorials and a market glossary for educational support.
-User-friendly UI with collapsible sections and metric highlights.

Installation
-pip install streamlit yfinance plotly scikit-learn pandas numpy

Run the dashboard locally with:
-streamlit run app.py

Code Structure
-fetch_live_stock_data(): Retrieves daily stock data for one year using Yahoo Finance.
-get_live_price(): Gets the latest closing stock price.
-get_prediction_models(): Trains four models on historical closing prices.
-generate_future_predictions(): Predicts closing prices for future days.
-generate_stock_plots(): Creates Plotly charts for exploration.
-main(): Handles Streamlit UI, input selection, and rendering charts & info.

Key Points
-The charts are interactive and responsive.
-The ML models help visualize potential future price trends, useful for analysis but not financial advice.
-The app is easily extensible for additional stocks, indicators, or data sources.
