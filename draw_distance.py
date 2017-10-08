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
path = '/home/zhangyan/triplet_loss'

chrN = 18
subImage_size = 100
chrN_start = chrN
chrN_end = chrN
anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)




prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")
pos_dis = np.zeros((output_pos_npy.shape[0]))
neg_dis = np.zeros((output_pos_npy.shape[0]))
K562_dis = np.zeros((output_pos_npy.shape[0]))
for i in range(0, output_anchor_npy.shape[0],1): 
    pos_dis[i] = d(output_anchor_npy[i], output_pos_npy[i])
    neg_dis[i] = d(output_anchor_npy[i], output_neg_npy[i])
    K562_dis[i] = d(output_anchor_npy[i], output_K562_npy[i])



print 'shape of our approach', output_pos_npy.shape, K562_dis.shape
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))

pixel_pos = np.zeros(HiC_anchor.shape[0])
pixel_neg = np.zeros(HiC_anchor.shape[0])
pixel_K562 = np.zeros(HiC_anchor.shape[0])
for i in range(0, HiC_anchor.shape[0] - subImage_size-1):
    pixel_pos[i] = np.sum(np.abs(positive[i][0] - anchor[i][0]))
    pixel_neg[i] = np.sum(np.abs(negative[i][0] - anchor[i][0]))
    pixel_K562[i] = np.sum(np.abs(K562[i][0] - anchor[i][0]))
print "pixel wise distance shape", pixel_pos.shape


Pearson_pos = np.load(path + '/pearson_pos_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_neg = np.load(path + '/pearson_neg_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_K562 = np.load(path + '/pearson_K562_vs_anchor_size100_chr'+str(chrN)+'.npy')
print 'Pearson', Pearson_pos.shape


start = 2850    
end = 3350
x = range(start,end)




plt.figure(figsize=(10, 3))

ax1 = plt.subplot(3, 1, 1) 
plt.plot(x, Pearson_pos[start-50:end-50], color='k', label='the same cell type',linewidth=3)
plt.plot(x, Pearson_K562[start-50:end-50], color='r', label='different cell types',linewidth=3)
#plt.axis('off')
plt.yticks([0.9, 0.95, 1])
plt.xticks([], [])
plt.xlim(start, end)
plt.legend(prop={'size':11}, bbox_to_anchor=(0.4, 0.0), loc=3,frameon=False)
#plt.ylim(0, 10)
#plt.grid()
#plt.show()
#plt.savefig('Pearson_sample_GM12878_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight', lw=0)
#plt.close() 

#plt.figure(figsize=(10, 1.5))
ax1 = plt.subplot(3, 1, 2) 
plt.plot(x, pixel_pos[start-50:end-50], color='k', label='the same cell type',linewidth=3)
plt.plot(x, pixel_K562[start-50:end-50], color='r', label='different cell types',linewidth=3)
#plt.grid()
plt.xlim(start, end)
plt.xticks([], [])
plt.yticks([10000, 30000, 50000])
plt.legend(prop={'size':11}, bbox_to_anchor=(0.4, 0), loc=3,frameon=False)
#plt.ylim(0, 10)
#plt.colorbar()
#plt.show()
#plt.savefig('pixel_wise_sample_K562_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight', lw=0)
#plt.close() 

#plt.figure(figsize=(10, 1.5))
ax1 = plt.subplot(3, 1, 3) 
plt.plot(x, pos_dis[start-50:end-50], color='k', label='the same cell type',linewidth=3)
plt.plot(x, neg_dis[start-50:end-50], color='r', label='different cell types',linewidth=3)
#plt.grid()
plt.yticks([0, 20, 40])
plt.legend(prop={'size':11}, bbox_to_anchor=(0.4, 0), loc=3,frameon=False)
plt.xlim(start, end)
#plt.ylim(0, 10)
#plt.colorbar()
#plt.show()
plt.savefig('distance_compare_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight', lw=0)
plt.close() 


 
