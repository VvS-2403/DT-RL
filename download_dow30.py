import yfinance as yf
import pandas as pd
import numpy as np

dow_30_tickers = [
    'AAPL', 'MSFT', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'CVX', 
    'MRK', 'KO', 'DIS', 'CSCO', 'MCD', 'CRM', 'VZ', 'NKE', 'AXP', 'HON', 
    'IBM', 'AMGN', 'BA', 'CAT', 'GS', 'MMM', 'INTC', 'WBA', 'TRV', 'DOW'
]

print("Downloading Dow 30 data (2013-2022)...")
df_stocks = yf.download(dow_30_tickers, start="2013-01-01", end="2022-12-31", group_by='ticker', auto_adjust=True)

records = []
for ticker in dow_30_tickers:
    if ticker in df_stocks.columns.levels[0]:
        df_ticker = df_stocks[ticker].copy()
        df_ticker.reset_index(inplace=True)
        df_ticker['Ticker'] = ticker
        df_ticker = df_ticker[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        records.append(df_ticker)
raw_df = pd.concat(records, ignore_index=True)

print("Downloading Macro indicators...")
macro_tickers = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY"}
macro_df = yf.download(list(macro_tickers.keys()), start="2013-01-01", end="2022-12-31", auto_adjust=True)

macro_records = pd.DataFrame(index=macro_df.index)
for ticker, name in macro_tickers.items():
    if isinstance(macro_df.columns, pd.MultiIndex):
        macro_records[name] = macro_df['Close'][ticker] if ('Close', ticker) in macro_df.columns else 0
    else:
        macro_records[name] = macro_df[ticker] if ticker in macro_df.columns else 0
macro_records.reset_index(inplace=True)

raw_df = pd.merge(raw_df, macro_records, on='Date', how='left')
raw_df[['VIX', 'TNX', 'DXY']] = raw_df[['VIX', 'TNX', 'DXY']].fillna(method='ffill').fillna(0)
raw_df["Sentiment_Score"] = np.random.uniform(-1, 1, len(raw_df))
raw_df = raw_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

print(f"Data shape: {raw_df.shape}")

output_file = "dow30_market_data_2013_2022.csv"
raw_df.to_csv(output_file, index=False)
print(f"Saved dataset to {output_file}")
