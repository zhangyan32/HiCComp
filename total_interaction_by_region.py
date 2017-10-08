
import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip

def calculateTotal(matrix, distance):
	result = np.zeros(matrix.shape[0])
	tempMatrix = np.zeros(matrix.shape)

	for i in range(0, matrix.shape[0] - 1):
		tempMatrix[i][i+1] = matrix[i][i+1]
		#print tempMatrix[i][i+1]
	for d in range(2, distance+1):
		#print d
		for i in range(0, matrix.shape[0] - d):
			tempMatrix[i][i+d] = tempMatrix[i+1][i+d] + tempMatrix[i][i+d-1] - tempMatrix[i+1][i+d-1] + matrix[i][i+d]
			if (d == distance):
				result[i + distance/2] = tempMatrix[i][i+d]
	return result

path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
distance = 100
result = calculateTotal(HiC_anchor, distance)
print np.mean(result)
np.save(path + '/GM12878_primary_total_interaction_underdistance_' + str(distance), result/np.mean(result))
result = calculateTotal(HiC_pos, distance)
print np.mean(result)
np.save(path + '/GM12878_replicate_total_interaction_underdistance_' + str(distance), result/np.mean(result))
result = calculateTotal(HiC_neg, distance)
print np.mean(result)
np.save(path + '/IMR90_total_interaction_underdistance_' + str(distance), result/np.mean(result))
result = calculateTotal(HiC_K562, distance)
print np.mean(result)
np.save(path + '/K562_total_interaction_underdistance_' + str(distance), result/np.mean(result))
