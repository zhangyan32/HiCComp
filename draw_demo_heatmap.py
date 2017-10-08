import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip
from scipy import ndimage

chrN = 18
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
#HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))

index = [[],[]]
for i in range(0, 500):
	for j in range(i+1, 500):
		index[0].append(i)
		index[1].append(j)
		if (j - i > 200):
			index[0].append(j)
			index[1].append(i)			

start = 2000   
end = 2500


sample = HiC_pos[start:end, start:end]
sample[index] = 0
sample = ndimage.rotate(sample, 45)
plt.imshow(sample,  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin= 0, vmax = 100)
plt.axis('off')
#plt.colorbar()
plt.savefig('hic_sample_pos_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')
plt.close() 


sample = HiC_anchor[start:end, start:end]
sample[index] = 0
sample = ndimage.rotate(sample, 45)
plt.imshow(sample,  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin= 0, vmax = 100)
plt.axis('off')
#plt.colorbar()
plt.savefig('hic_sample_anchor_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')
plt.close() 


sample = HiC_neg[start:end, start:end]
sample[index] = 0
sample = ndimage.rotate(sample, 45)
plt.imshow(sample,  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin= 0, vmax = 100)
plt.axis('off')
#plt.colorbar()
plt.savefig('hic_sample_neg_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')
plt.close() 