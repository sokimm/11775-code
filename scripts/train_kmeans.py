#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
import cPickle
import sys
import pandas

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])	
    
    mfcc_array = pandas.read_csv(mfcc_csv_file, header = None, sep = ';', dtype='float64')
    kmeans = KMeans(n_clusters=cluster_num).fit(mfcc_array)

    # finally save the k-means model
    cPickle.dump(kmeans, open(output_file,"wb"))#, cPickle.HIGHEST_PROTOCOL)
 
    print "K-means trained successfully!"
