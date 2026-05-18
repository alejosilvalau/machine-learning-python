import os

import pandas as pd
import pandas_ta as ta
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.linear_model import LinearRegression

load_dotenv()
_ = ta.version  # Just to ensure the extension is loaded and available.


# 1. DATA GATHERING & FEATURES
def get_stock_data(symbol):
    df = yf.download(symbol, period="4mo", interval="1d")

    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate Indicators using the extension
    df["RSI"] = df.ta.rsi(length=14)

    macd_df = df.ta.macd()
    df["MACD"] = macd_df["MACD_12_26_9"]

    df["Volume_Norm"] = df["Volume"] / df["Volume"].max()

    # Target: Tomorrow's price
    df["Target"] = df["Close"].shift(-1)

    # Filter to last 3 months and clean
    three_months_ago = df.index[-1] - pd.DateOffset(months=3)
    df = df[df.index >= three_months_ago].dropna()
    return df


# 2. MODEL TRAINING & PREDICTION
def run_predictions(df):
    features = ["Close", "Volume_Norm", "RSI", "MACD"]
    X = df[features].values
    y = df["Target"].values

    model = LinearRegression()
    model.fit(X, y)

    # Get the most recent data point to predict the future
    latest_data = df[features].iloc[-1:].values

    # For a simple MVP, we project the daily trend for longer horizons
    next_day = model.predict(latest_data)[0]

    # Calculating the "Delta" (the expected daily change)
    current_price = df["Close"].iloc[-1]
    daily_delta = next_day - current_price

    predictions = {
        "current": round(current_price, 2),
        "tomorrow": round(next_day, 2),
        "week": round(current_price + (daily_delta * 7), 2),
        "month": round(current_price + (daily_delta * 30), 2),
        "year": round(current_price + (daily_delta * 365), 2),
    }
    return predictions


# 3. LANGCHAIN INTERPRETER
def explain_with_ai(symbol, preds):
    github_endpoint = "https://models.github.ai/inference"

    llm = ChatOpenAI(
        base_url=github_endpoint,
        api_key=os.getenv("GITHUB_TOKEN"),
        model="gpt-4.1-mini",
        temperature=0.7,
    )

    template = """
    You are a financial analyst assistant. A Linear Regression model has predicted the following 
    prices for {symbol} based on the last 3 months of RSI, MACD, and Volume data:
    
    - Current Price: ${current}
    - Tomorrow: ${tomorrow}
    - 1 Week: ${week}
    - 1 Month: ${month}
    - 1 Year: ${year}
    
    Briefly explain these results to the user. Mention that Linear Regression assumes current 
    trends continue and warn about the volatility of long-term predictions (1 year). 
    Keep it professional but accessible for a student project.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({**preds, "symbol": symbol})


# 4. TERMINAL INTERFACE
def main():
    print("=== Stock Prediction MVP ===")
    symbol = input("Enter Stock Symbol (e.g., AAPL): ").upper()

    try:
        print(f"Fetching data and training model for {symbol}...")
        df = get_stock_data(symbol)
        preds = run_predictions(df)

        print("\n--- Model Results ---")
        response = explain_with_ai(symbol, preds)
        print(response.content)

    except Exception as e:
        print(f"Error: {e}. Make sure the symbol is correct.")


if __name__ == "__main__":
    main()
