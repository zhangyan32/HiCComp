import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip

from scipy.stats import pearsonr
from scipy.stats import spearmanr

path = '/home/zhangyan/triplet_loss'
for n in range(0, 13):
    epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,n]
    epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,n]
    print n
    #bin_diff = np.load('state_diff_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
    #bin_union = np.load('state_share_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
    #different = bin_diff / (bin_union + 0.001)

    start = 2000   
    end = 2500
    x = range(start,end)
    plt.figure(figsize=(10, 2))
    ax1 = plt.subplot(2, 1, 1) 

    plt.bar(x, epi_GM12878[start:end], color='k')
    #plt.axis('off')
    plt.xlim(start, end)
    plt.ylim(0, 10)
    plt.xticks([], [])
    #plt.grid()
    #plt.colorbar()
    #plt.show()
    #plt.savefig('CTCF_sample_GM12878_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight', lw=0)
    #plt.close() 
    ax1 = plt.subplot(2, 1, 2)
    #plt.figure(figsize=(10, 2))
    #sample = 
    plt.bar(x, epi_K562[start:end] , color='k')
    #plt.grid()
    plt.xlim(start, end)
    #plt.xticks([], [])
    plt.ylim(0, 10)
    #plt.colorbar()
    #plt.show()
    print n
    plt.savefig('Chipseq_sample_compare_xticks_'+str(start)+'_'+str(end)+'_'+str(n)+'.png', bbox_inches='tight', lw=0)
    plt.close() 




 
