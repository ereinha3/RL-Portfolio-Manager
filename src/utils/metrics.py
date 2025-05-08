import numpy as np

def summary_stats(series):
    ret = series.pct_change().dropna()
    sharpe = ret.mean()/ret.std()*np.sqrt(252)
    mdd = (series/series.cummax()-1).min()
    return {'TotalRet':series.iloc[-1]/series.iloc[0]-1, 'Sharpe':sharpe, 'MaxDD':mdd}