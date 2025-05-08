import matplotlib.pyplot as plt

def plot_cum(ser, title="Cumulative"):
    ser.div(ser.iloc[0]).plot(); plt.title(title); plt.show()