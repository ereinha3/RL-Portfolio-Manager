import numpy as np
import pandas as pd
from config import WINDOW

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    # prices: MultiIndex (Ticker, OHLCV)
    # extract closes
    closes = prices['Close']
    lr = np.log(closes / closes.shift(1)).dropna(how='all')

    sma20 = closes.rolling(WINDOW).mean()
    rsi14 = closes.rolling(WINDOW).apply(
        lambda x: ((x.diff()[1:] > 0).sum() / WINDOW) * 100, raw=False
    )
    vol20 = lr.rolling(WINDOW).std()

    # stack features: shape (T, N*3)
    feats = pd.concat([
        lr.add_suffix('_lr'),
        (closes - sma20).divide(sma20).add_suffix('_ma20_gap'),
        rsi14.add_suffix('_rsi14'),
        vol20.add_suffix('_vol20')
    ], axis=1).dropna(how='all')

    return feats