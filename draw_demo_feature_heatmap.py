import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip
from scipy import ndimage
path = '/home/zhangyan/triplet_loss'
chrN = 18
#Perason_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_substraction_important_samples.npy')
#Pixel_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples_nan_distance_mean.npy')


Perason_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_substraction_Rep1_important_samples.npy') 




Pixel_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_substraction_Rep2_important_samples.npy')


CNN_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples.npy')
start = 2000   
end = 4000
'''
print (Pixel_map > 0)
#sample = ndimage.rotate(sample, 45)
plt.imshow(Pixel_map[start:end, start:end],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin= 0, vmax=1)
plt.axis('off')
plt.colorbar()
plt.show()
'''


'''
index = [[],[]]
for i in range(0, Pixel_map.shape[0]):
    for j in range(i+1, Pixel_map.shape[0]):
    	if (j - i > 100):
    		index[0].append(j)
        	index[1].append(i)
        index[0].append(i)
        index[1].append(j)
#Perason_map = np.nan_to_num(Perason_map)
#Pixel_map = np.nan_to_num(Pixel_map)
#CNN_map = np.nan_to_num(CNN_map)

Perason_map[index] = float('nan')
Pixel_map[index] = float('nan')
CNN_map[index] = float('nan')		
'''
start = 2000   
end = 2500


sample = Perason_map[start:end, start:end]
#print sample
#sample[index] = 0
#sample = ndimage.rotate(sample, 45, reshape=False)
#print sample
plt.imshow(sample,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
plt.axis('off')
#plt.colorbar()
plt.savefig('DirectSubstractrep1_map_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')

plt.close() 


sample = Pixel_map[start:end, start:end]
#sample[index] = 0
#sample = ndimage.rotate(sample, 45)
plt.imshow(sample,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
plt.axis('off')
plt.colorbar()
plt.savefig('DirectSubstractrep2_map_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')
plt.close() 


sample = CNN_map[start:end, start:end]
#sample[index] = 0
#sample = ndimage.rotate(sample, 45)
plt.imshow(sample,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
plt.axis('off')
#plt.colorbar()
#plt.savefig('CNN_map_'+str(start)+'_'+str(end)+'.png', bbox_inches='tight')
plt.close() 