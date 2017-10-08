import numpy as np
import copy

import matplotlib.pyplot as plt
import os
def readLoop(path, chrN, min_size = 0, max_size = 500, resolution = 10000):
    infile = open(path).readlines()
    looplist = []
    for j in range(1, len(infile)):
        line = infile[j]
        words = line.split()
        aLoop = []
        
        if ('X' in words[0]):
            words[0] = '23'
        if ('X' in words[3]):
            words[3] = '23'
        if (words[0] != words[3]):
            print "interchrome..."
            continue
        if (int(words[0])!=chrN):
            continue
        aLoop.append(int(words[0]))
        start = int(words[1])/resolution
        end = int(words[4])/resolution
        size = abs(end - start)
        if (size < min_size or size > max_size):
            continue
        if (end < start):
            aLoop.append(end)
            aLoop.append(start)
        else:
            aLoop.append(start)
            aLoop.append(end)
        looplist.append(aLoop)
    print "length of looplist is ", len(looplist)
    return looplist

path = '/home/zhangyan/triplet_loss'
chrN = 18
chrN_start = chrN     
chrN_end = chrN
subImage_size = 100
entire_looplist = readLoop('/home/zhangyan/looplist/GSE63525_GM12878_primary_HiCCUPS_looplist.txt' , chrN)

def count_loop(looplist, location):
    count = 0
    for loop in looplist:
        if (location < loop[1] and location + 100 > loop[2]):
            count += 1
    return count
labels = np.zeros(7807)

for i in range(0, labels.shape[0]):
    labels[i] = count_loop(entire_looplist, i)




pos_index = []
neg_index = []

for i in range(2, labels.shape[0]-1000, 100


    ):
    if labels[i] > 0 :
        pos_index.append(i)
    if (labels[i] <= 0):
        neg_index.append(i)

print len(pos_index), len(neg_index)
print "base is ", len(pos_index)/(1.0 * len(neg_index) + len(pos_index))

print np.count_nonzero(labels)

path = '/home/zhangyan/triplet_loss'
chrN = 18
chrN_start = chrN     
chrN_end = chrN
subImage_size = 100
anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
#negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
#K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)

prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
#output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
#output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
#output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")

X = np.concatenate((output_anchor_npy[pos_index], output_anchor_npy[neg_index]), axis = 0)
RawHiC = anchor.reshape((anchor.shape[0], -1))  
print "shape of RawHiC", RawHiC.shape
X2 = np.concatenate((RawHiC[pos_index], RawHiC[neg_index]), axis = 0)
y = np.zeros(X.shape[0])
y[:len(pos_index)] = 1
print X.shape, y.shape

from sklearn import datasets

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import datasets, linear_model

clf = linear_model.LogisticRegression()
scores = cross_val_score(clf, X, y, cv=3)
print scores      
print len(scores)                                        

print("Use learned features Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = linear_model.LogisticRegression()
scores = cross_val_score(clf, X2, y, cv=3)
print scores      
print len(scores)                                        

print("Use Raw HiC Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))









for i in range(100000, 7807):
    if labels[i] != 0:
        continue
    print i, labels[i]
    for loop in entire_looplist:
        if (loop[2] < i + 100 and loop[1] > i):
            print loop
    plt.figure(figsize=(30, 30))
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text(str(i))
    plt.imshow(anchor[i][0], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(1, 2, 2)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(positive[i][0], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()
    plt.show()



