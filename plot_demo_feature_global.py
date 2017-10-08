import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip
from scipy import ndimage
path = '/home/zhangyan/triplet_loss'
chrN = 18

for latent_index in range(0, 1):
	#CNN_map = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_'+str(latent_index)+'.npy')
	CNN_map = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_all_latent.npy')


	start = 2000   
	end = 2500


	sample = CNN_map[start:end, start:end]
	#sample[index] = 0
	#sample = ndimage.rotate(sample, 45)
	plt.imshow(sample,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
	plt.axis('off')
	plt.colorbar()
	#plt.savefig('CNN_map_'+str(start)+'_'+str(end)+'_' +str(latent_index)+ '.png', bbox_inches='tight')
	plt.savefig('CNN_map_'+str(start)+'_'+str(end)+'_all_latent.png', bbox_inches='tight')
	plt.close() 