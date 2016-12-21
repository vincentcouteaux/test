import numpy as np
import matplotlib.pyplot as plt
import csv

class City:
    def __init__(self, index):
        self.stations = {}
        self.index = index

    def display(self):
        for k, v in self.stations.iteritems():
            print("Station {0}: ".format(k))
            v.display()

class Stations:
    def __init__(self, index):
        self.index = index
        self.hlres = {}
        self.route = {}
        self.hldres = {}
        self.port = 0
        self.industry = 0
        self.natural = 0
        self.roadinvdist = 0
        self.green = 0
        self.time_series = {}
    def display(self):
        print('* hlres : {0}'.format(self.hlres))
        print('* hldres : {0}'.format(self.hldres))
        print('* route : {0}'.format(self.route))
        print('* port : {0}'.format(self.port))
        print('* industry : {0}'.format(self.industry))
        print('* natural : {0}'.format(self.natural))
        print('* roadinvdist : {0}'.format(self.roadinvdist))
        print('* green : {0}'.format(self.green))

class TimeSerie:
    def __init__(self, polutant):
        self.polutant = polutant
        self.time = []
        self.cloud = []
        self.calm_day = []
        self.temperature = []
        self.windcos = []
        self.windsin = []
        self.windint = []
        self.precint = []
        self.precprob = []
        self.pressure = []
        self.db_index = []
        self.value = []

def str2float(string):
    if string:
        return float(string)
    else:
        return 0.

def parse(filename, yname=''):
    cities = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        columns_list = reader.next()
        columns = {}
        for i, k in enumerate(columns_list):
            columns[k] = i
        print(columns)
        for row in reader:
            current_city = int(float(row[columns['zone_id']]))
            if current_city not in cities:
                cities[current_city] = City(current_city)
            current_station = int(float(row[columns['station_id']]))
            if current_station not in cities[current_city].stations:
                cities[current_city].stations[current_station] = Stations(current_station)
                station = cities[current_city].stations[current_station]
                station.hlres[50] = str2float(row[columns['hlres_50']])
                station.hlres[100] = str2float(row[columns['hlres_100']])
                station.hlres[300] = str2float(row[columns['hlres_300']])
                station.hlres[500] = str2float(row[columns['hlres_500']])
                station.hlres[1000] = str2float(row[columns['hlres_1000']])
                station.hldres[50] = str2float(row[columns['hldres_50']])
                station.hldres[100] = str2float(row[columns['hldres_100']])
                station.hldres[300] = str2float(row[columns['hldres_300']])
                station.hldres[500] = str2float(row[columns['hldres_500']])
                station.hldres[1000] = str2float(row[columns['hldres_1000']])
                station.route[100] = str2float(row[columns['route_100']])
                station.route[300] = str2float(row[columns['route_300']])
                station.route[500] = str2float(row[columns['route_500']])
                station.route[1000] = str2float(row[columns['route_1000']])
                station.port = str2float(row[columns['port_5000']])
                station.industry = str2float(row[columns['industry_1000']])
                station.natural = str2float(row[columns['natural_5000']])
                station.roadinvdist = str2float(row[columns['roadinvdist']])
                station.green = str2float(row[columns['green_5000']])
            station = cities[current_city].stations[current_station]
            current_polutant = row[columns['pollutant']]
            if current_polutant not in station.time_series:
                station.time_series[current_polutant] = TimeSerie(current_polutant)
            timeSerie = station.time_series[current_polutant]
            timeSerie.time.append(float(row[columns['daytime']]))
            timeSerie.cloud.append(float(row[columns['cloudcover']]))
            timeSerie.calm_day.append(row[columns['is_calmday']])
            timeSerie.temperature.append(float(row[columns['temperature']]))
            timeSerie.windcos.append(float(row[columns['windbearingcos']]))
            timeSerie.windsin.append(float(row[columns['windbearingsin']]))
            timeSerie.windint.append(float(row[columns['windspeed']]))
            timeSerie.precint.append(float(row[columns['precipintensity']]))
            timeSerie.precprob.append(float(row[columns['precipprobability']]))
            timeSerie.pressure.append(float(row[columns['pressure']]))
            timeSerie.db_index.append(int(row[columns['ID']]))
    return cities

if __name__ == "__main__":
    cities = parse('X_train.csv')
    cities[0].display()
    cities[1].display()
    cities[2].display()
    x = np.array(cities[0].stations[16].time_series['NO2'].time)
    t = np.array(cities[0].stations[16].time_series['NO2'].temperature)
    p = np.array(cities[0].stations[16].time_series['NO2'].pressure)
    plt.plot(x, t)
    plt.plot(x, p)
    plt.show()


