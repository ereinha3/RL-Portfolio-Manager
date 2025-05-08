#!/usr/bin/env python3
"""
fetch_intraday.py

Fetches 1-hour OHLCV data for:
  • S&P 500 constituents (from GitHub CSV)
  • NASDAQ-100 constituents (from Wikipedia)
  • A basket of high-volume ETFs

Automatically removes any tickers that failed (all-NaN),
and saves the cleaned panel plus the final ticker list.
"""

import os
import pandas as pd
import yfinance as yf

RAW_DIR   = os.path.join(os.getcwd(), 'data', 'raw')
OUT_PARQ  = os.path.join(RAW_DIR, 'prices_1h.parquet')
OUT_CSV   = os.path.join(RAW_DIR, 'prices_1h.csv')
OUT_TICKS = os.path.join(RAW_DIR, 'tickers.txt')

def get_sp500_tickers() -> list[str]:
    url = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies/main/data/constituents.csv")
    return pd.read_csv(url)['Symbol'].dropna().tolist()

def get_nasdaq100_tickers() -> list[str]:
    wiki_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(wiki_url, header=0)
    for tbl in tables:
        if 'Ticker' in tbl.columns:
            return tbl['Ticker'].dropna().astype(str).tolist()
    raise RuntimeError("NASDAQ-100 table not found on Wikipedia")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    sp500  = get_sp500_tickers()
    nas100 = get_nasdaq100_tickers()
    etfs   = ['SPY','QQQ','IWM','TLT','GLD','HYG','XLF','XLK','XLV','XLY']
    universe = list(dict.fromkeys(sp500 + nas100 + etfs))
    print(f"⏳ Fetching {len(universe)} tickers at 1h resolution...")

    df = yf.download(
        tickers=universe,
        period='730d',       # up to 2 yrs of 60m bars
        interval='60m',
        auto_adjust=True,    # splits & dividends
        group_by='ticker',
        threads=True
    )

    tickers = df.columns.get_level_values(0).unique()
    dropped = []
    for t in tickers:
        sub = df[t]
        if sub.isna().all().all():
            dropped.append(t)

    if dropped:
        print(f"Dropping {len(dropped)} failed tickers: {dropped}")
        df = df.drop(labels=dropped, axis=1, level=0)
        universe = [t for t in universe if t not in dropped]

    df.to_parquet(OUT_PARQ)
    print(f"Saved cleaned intraday data to {OUT_PARQ}")

    df.to_csv(OUT_CSV)
    print(f"Saved cleaned intraday data (for viewing) to {OUT_CSV}")

    with open(OUT_TICKS, 'w') as f:
        for t in universe:
            f.write(t + '\n')
    print(f"Final universe ({len(universe)} tickers) written to {OUT_TICKS}")

if __name__ == '__main__':
    main()
