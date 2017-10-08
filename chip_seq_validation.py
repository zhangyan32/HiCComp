import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip

from scipy.stats import pearsonr
from scipy.stats import spearmanr
def d(M1, M2):
    return np.sqrt(np.sum((M1 - M2) ** 2))
epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,0]

epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,0]
print epi_K562.shape, epi_GM12878.shape
print epi_GM12878, epi_K562
epi_distance = np.sqrt((epi_K562 - epi_GM12878) * (epi_K562 - epi_GM12878))


epi_distance /= (epi_GM12878 + epi_K562 + 0.0001)
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
for i in range(subImage_size/2, output_anchor_npy.shape[0],1): 
    pos_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_pos_npy[i-subImage_size/2])
    neg_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_neg_npy[i-subImage_size/2])
    K562_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_K562_npy[i-subImage_size/2])

epi_distance[0:subImage_size/2] = 0
epi_distance[-subImage_size/2:] = 0
print pearsonr(K562_dis - pos_dis, epi_distance)
plt.scatter(K562_dis/(pos_dis + 0.001), epi_distance)
#plt.ylim(0, 10)
plt.show()




print epi_distance.shape