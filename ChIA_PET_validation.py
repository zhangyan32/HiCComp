import numpy as np
import copy

import matplotlib.pyplot as plt
import os
import gzip
subImage_size = 100
ChIAPET = np.load(gzip.GzipFile('/home/zhangyan/ChIA_PET/GM12878_CTCF_18.npy.gz', "r"))
print ChIAPET.shape
print np.count_nonzero(ChIAPET)
labels = np.zeros(ChIAPET.shape[0]-subImage_size)
for i in range(0, ChIAPET.shape[0] - subImage_size):
	labels[i] = np.count_nonzero(ChIAPET[i+5:i+subImage_size-5, i+5:i+subImage_size-5])

path = '/home/zhangyan/triplet_loss'

chrN = 18
subImage_size = 100
chrN_start = chrN
chrN_end = chrN

prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")

pos_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
neg_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
K562_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))


print output_K562_npy.shape, labels.shape
print np.count_nonzero(labels)
from sklearn import datasets, linear_model

clf = linear_model.LinearRegression()
clf.fit(output_K562_npy, labels[0:7707])
prediction = clf.predict(output_K562_npy)
print prediction

plt.scatter(labels[0:7707], prediction)
plt.show()