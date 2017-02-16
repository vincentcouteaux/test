import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

csv.field_size_limit(sys.maxsize)

def retrieve_all_train():
    filename="Xtr.csv"
    images = np.zeros((5000, 3072))
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        c = 0
        for row in reader:
            images[c, :] = row[:3072]
            c += 1
    return images

def retrieve_labels():
    filename="Ytr.csv"
    labels = np.zeros(5000)
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        c = 0
        reader.next()
        for row in reader:
            labels[c] = row[1]
            c += 1
    return labels

if __name__ == "__main__":
    images = retrieve_all_train()
    images = np.reshape(images, (-1, 32, 32, 3), "F")
    images = np.swapaxes(images, 1, 2)
    labels = retrieve_labels()
    for k in np.where(labels == 5)[0][:10]:
        plt.figure()
        plt.imshow(images[k])
        plt.title(labels[k])
    plt.show()
