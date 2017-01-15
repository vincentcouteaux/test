import numpy as np
import csv

def mape(forecast, truth):
    return np.mean(np.abs((forecast-truth)/truth))

def test_to_csv(forecast, filename):
    with open(filename + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        spamwriter.writerow(['ID', 'TARGET'])
        for i in range(forecast.size):
            spamwriter.writerow([i, forecast[i]])

