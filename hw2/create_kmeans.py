#!/bin/python
import numpy as np
import os
import pickle
from sklearn.preprocessing import normalize
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector
 
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print ("kmeans_model -- path to the kmeans model")
        print ("cluster_num -- number of cluster")
        print ("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = pickle.load(open(kmeans_model,"rb"))

    fread = open(file_list, "r")
    for line in fread.readlines():
        surf_path = "surf/" + line.replace('\n','') + ".pkl"
        fwrite = open('kmeans_' + str(cluster_num) + "/" +  line.replace('\n',''),'w')
        cluster_histogram = np.zeros(cluster_num)
    
        if os.path.exists(surf_path) == True:
            try:            
                array = pickle.load(open(surf_path, "rb"))
    
                for n in range(len(array)):
                    if array[n].shape == () or array[n].shape == (1,):
                        continue
                    feature = array[n].reshape(-1, 64)
                    feat_len, feat_dim = feature.shape
    
                np.random.shuffle(feature)
                if feat_len > cluster_num*4:
                    feature = feature[:cluster_num*4,:]
    
                feature = normalize(feature, axis=1)
    
                if n == 0:
                    feature_ = feature.copy()
                else:
                    feature_ = np.vstack((feature_, feature))
    
                pred = kmeans.predict(array)
                pred_count = np.bincount(pred, minlength=cluster_num)
            except:
                pred_count = np.bincount(np.array(range(cluster_num)), minlength=cluster_num)
        else:
            pred_count = np.bincount(np.array(range(cluster_num)), minlength=cluster_num)
    
        cluster_histogram = pred_count.astype(float)/cluster_num
        cluster_histogram = cluster_histogram/sum(cluster_histogram)
    
        line = str(cluster_histogram[0])
        for m in range(1, cluster_num):
            line += ';' + str(cluster_histogram[m])
            fwrite.write(line + '\n')
        fwrite.close()

    print("K-means features generated successfully!")
