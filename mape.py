import numpy as np

def mape(forecast, truth):
    return np.mean(np.abs((forecast-truth)/truth))
