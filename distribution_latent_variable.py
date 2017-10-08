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

'''
bin_diff = np.load('state_diff_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
bin_union = np.load('state_share_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
pos_index = []
neg_index = []
for i in range(subImage_size/2, bin_diff.shape[0]-subImage_size/2-1):
    if (bin_diff[i] == bin_union[i] and bin_union[i] >= 1000):
        pos_index.append(i-subImage_size/2)
        continue
    if (bin_diff[i] == 0 and bin_union[i] >= 1000):
        neg_index.append(i-subImage_size/2)

print "number of positive", len(pos_index)
print "number of negative", len(neg_index)
'''
pos_index = []
neg_index = []

epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,0]
epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,0]
upper = 2
lower = 1
for i in range(subImage_size/2, epi_GM12878.shape[0]-subImage_size/2-1):
    if (epi_GM12878[i] >= 2 and epi_K562[i] < 1):
        pos_index.append(i-subImage_size/2)
    elif (epi_K562[i] >= 2 and epi_GM12878[i] < 1):
        pos_index.append(i-subImage_size/2)
    elif (epi_GM12878[i] >= 2 and epi_K562[i] >= 2):
        neg_index.append(i-subImage_size/2)


print "number of positive", len(pos_index)
print "number of negative", len(neg_index)


'''
entire_looplist = np.load('/home/zhangyan/looplist/GM12878_and_K562_loop.npy' ).item()
print entire_looplist
pos_index = []
neg_index = []


for key in entire_looplist.keys():
    loops = entire_looplist[key]
    if ((key[1] + key[2])/2 >= 7707):
        continue
    #if (1 in loops and 3 in loops):
    #    neg_index.append((key[1] + key[2])/2)
    if (1 in loops and (3 not in loops and not 2 in loops)):
        pos_index.append((key[1] + key[2])/2)        
    else:
        neg_index.append((key[1] + key[2])/2)

print "number of positive", len(pos_index)
print "number of negative", len(neg_index)
'''


anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)

prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")

print "output shape", output_neg_npy.shape
for i in range(0, 7):
    print i
    plt.figure(figsize=(30, 30))

    ax1 = plt.subplot(1, 3, 1)
    ax1.title.set_text('pos' )
    plt.hist(output_pos_npy[:, i], bins = 100)
    plt.grid()

    ax1 = plt.subplot(1, 3, 2)
    ax1.title.set_text('neg' )
    plt.hist(output_neg_npy[:, i], bins = 100)
    plt.grid()

    ax1 = plt.subplot(1, 3, 3)
    ax1.title.set_text('K562' )
    plt.hist(output_K562_npy[:, i], bins = 100)
    plt.grid()

    plt.show()
    plt.close()


HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))






