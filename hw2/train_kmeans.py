#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
import pickle
import pandas
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} surf_csv_file cluster_num output_file".format(sys.argv[0]))
        print("surf_csv_file -- path to the surf csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    surf_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    surf_array = pandas.read_csv(surf_csv_file, header = None, sep = ';', dtype='float64')
    kmeans = KMeans(n_clusters=cluster_num).fit(surf_array)

    # finally save the k-means model
    pickle.dump(kmeans, open(output_file,"wb"))#, cPickle.HIGHEST_PROTOCOL)

    print( "K-means trained successfully!")
