#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.getcwd(), 'data', 'raw')
PARQ    = os.path.join(RAW_DIR, 'prices_1h.parquet')
TICKS   = os.path.join(RAW_DIR, 'tickers_final.txt')

# Reuse functions from previous fetch script
from config import TICKS

def get_sp500_tickers():
    url = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies/main/data/constituents.csv")
    return pd.read_csv(url)['Symbol'].dropna().tolist()

def get_nasdaq100_tickers():
    wiki = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(wiki)
    for tbl in tables:
        if 'Ticker' in tbl.columns:
            return tbl['Ticker'].astype(str).tolist()
    raise RuntimeError("NASDAQ-100 table not found")

if __name__ == '__main__':
    os.makedirs(RAW_DIR, exist_ok=True)
    sp500  = get_sp500_tickers()
    nasdaq = get_nasdaq100_tickers()
    etfs   = ['SPY','QQQ','IWM','TLT','GLD','HYG','XLF','XLK','XLV','XLY']
    universe = list(dict.fromkeys(sp500 + nasdaq + etfs))
    print(f"Fetching {len(universe)} tickers...")

    df = yf.download(
        tickers=universe,
        period='730d', interval='60m',
        auto_adjust=True, group_by='ticker', threads=True
    )

    # Drop failed (all-NaN) tickers
    cols = df.columns.get_level_values(0).unique()
    failed = [t for t in cols if df[t].isna().all().all()]
    for t in failed:
        universe.remove(t)
        df = df.drop(t, axis=1, level=0)
    print(f"Dropped {len(failed)} failed: {failed}")

    # Save
    df.to_parquet(PARQ)
    with open(TICKS, 'w') as f:
        f.write("\n".join(universe))
    print(f"Saved data to {PARQ}, tickers to {TICKS}")