#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
from nltk.tokenize import word_tokenize
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)
    
    vocab_file = sys.argv[1]; file_list = sys.argv[2]

    # read the vocabulary into memory
    vocab = []
    fopen = open(vocab_file, 'r')
    for line in fopen.readlines():
        vocab.append(line.replace('\n',''))
    fopen.close()

    vocab_size = len(vocab)
    fread = open(file_list, "r")
    for line in fread.readlines():
        # not flexible because we hardcode the asr and asrfeat directories here
        asr_path = "../asrs/" + line.replace('\n','') + ".txt"
        fwrite = open('asrfeat/' + line.replace('\n',''),'w')
        cluster_histogram = numpy.zeros(vocab_size)
        total_occur = 0

        if os.path.exists(asr_path) == True:
            fread = open(asr_path, 'r')
            for asr_line in fread.readlines():
                splits = word_tokenize(asr_line)
                for word in splits:
                    if word in vocab:
                        cluster_histogram[vocab.index(word)] += 1
                        total_occur += 1
            fread.close()

        if total_occur > 0:
            # normalize the histogram to be a probability distribution
            for m in xrange(vocab_size):
                cluster_histogram[m] /= float(total_occur)
        else:
            cluster_histogram.fill(1.0/vocab_size) # for videos that have no ASR features, we simply set all the values to be
                                                   # 1.0/vocab_size

        line = str(cluster_histogram[0])
        for m in range(1, vocab_size):
            line += ';' + str(cluster_histogram[m])
        fwrite.write(line + '\n')
        fwrite.close()

    print "ASR features generated successfully!"
