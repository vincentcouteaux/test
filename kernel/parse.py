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

def retrieve_first_train(n):
    """ retrieve n first lines in the train file"""
    filename="Xtr.csv"
    images = np.zeros((n, 3072))
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        c = 0
        for row in reader:
            images[c, :] = row[:3072]
            c += 1
            if c >= n:
                break
    images = np.reshape(images, (-1, 32, 32, 3), "F")
    images = normalize(np.swapaxes(images, 1, 2))
    return images

def normalize(images):
    out = np.zeros(images.shape)
    for i, im in enumerate(images):
        out[i] = (im - np.min(im))/(np.max(im) - np.min(im))
    return out

def traindb():
    images = retrieve_all_train()
    images = np.reshape(images, (-1, 32, 32, 3), "F")
    images = normalize(np.swapaxes(images, 1, 2))
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
    images = normalize(np.swapaxes(images, 1, 2))
    labels = retrieve_labels()
    for k in np.where(labels == 1)[0][:10]:
        plt.figure()
        plt.imshow(images[k], interpolation="nearest")
        plt.title(labels[k])
    plt.show()
