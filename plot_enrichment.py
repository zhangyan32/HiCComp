import numpy as np
import model

from time import gmtime, strftime
import gzip
import matplotlib.pyplot as plt
import sys
import random
from scipy import stats
#data = np.load('chip_seq_enrich_resample2.npy')
data = np.load('chip_seq_enrich_newsubstraction2.npy')
data = np.load('chip_seq_enrich_revised_feature_all_latent.npy')

CNN = np.zeros((2000, 13))
Pixel = np.zeros((2000, 13))
Pearson = np.zeros((2000, 13))
Neg = np.zeros((7808, 13))
#Neg = np.zeros((2000, 13))

for i in range(0, 13):
	CNN[:,i] = data[i, 0]
	Pixel[:,i] = data[i, 1]
	Pearson[:,i] = data[i, 2]
	Neg[:,i] = data[i, 3]


my_xticks = ['CTCF','DNase',"H2A.Z","H3K27ac",'H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K79me2','H3K9ac','H3K9me3','H4K20me1']



print CNN.shape

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
n = 7
pos = np.array(range(0, 13*n, n))
print pos.shape
ax.boxplot(CNN,  showfliers=False,notch=True,whis=0, boxprops=dict(linestyle='-', linewidth=2, color='red'), whiskerprops=dict(linestyle='-', linewidth=1, color='red'),  positions=pos, sym = "",  widths=1)

#ax.boxplot(CNN,  boxprops=dict(linestyle='-', linewidth=2, color='red'), whiskerprops=dict(linestyle='-', linewidth=1, color='red'),  notch=True,positions=pos, sym = '',  widths=1)
ax.boxplot(Pixel,  whis=0,boxprops=dict(linestyle='-', linewidth=2, color='green'), whiskerprops=dict(linestyle='-', linewidth=1, color='green'), notch=True,positions=pos+1.2, sym = '', widths=1, labels=range(0, 13*n, n))
ax.boxplot(Pearson, whis=0, boxprops=dict(linestyle='-', linewidth=2, color='blue'), whiskerprops=dict(linestyle='-', linewidth=1, color='blue'), notch=True,positions=pos+2.4, sym = '', widths=1, labels=range(0, 13*n, n))
ax.boxplot(Neg, whis=0, boxprops=dict(linestyle='-', linewidth=2, color='black'), whiskerprops=dict(linestyle='-', linewidth=1, color='black'), notch=True,positions=pos+3.6, sym = '', widths=1, labels=range(0, 13*n, n))
#ax.set_xlabel('Data Points')
ax.set_ylabel('Difference of enrichment', fontsize=16)
plt.xlim(-3, 90)
plt.ylim(0, 1)
hB, = plt.plot([2,2],'b-',linewidth=3)
hR, = plt.plot([2,2],'r-',linewidth=3)
hK, = plt.plot([2,2],'k-',linewidth=3)
hG, = plt.plot([2,2],'g-',linewidth=3)

plt.legend((hR, hG, hB, hK),('HiCComp', 'Matrix substraction rep1','Matrix substraction rep2', 'Average of entire chromosome'), prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
plt.xticks(pos, my_xticks, fontsize=16)
plt.yticks(fontsize = 16)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=60)
plt.savefig('Enrichment_revise_feature_all.png', bbox_inches='tight')
#plt.show()