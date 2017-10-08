import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip
from scipy import ndimage
path = '/home/zhangyan/triplet_loss'
chrN = 18
#Perason_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_substraction_important_samples.npy')
#Pixel_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples_nan_distance_mean.npy')

chrN = 18
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))


CNN_map_4 = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8.npy')
CNN_map_8 = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_windows_8.npy')
CNN_map_16 = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_windows_16.npy')
start = 2000   
end = 4000

start = 2000   
end = 2500
for start in range(0, 7000, 500):
    end = start + 500
    plt.figure()
    ax1 = plt.subplot(2, 3, 1)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 25)
    plt.grid()

    ax1 = plt.subplot(2, 3, 2)
    ax1.title.set_text('GM12878' )
    plt.imshow(HiC_pos[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 25)
    plt.grid()

    ax1 = plt.subplot(2, 3, 3)
    ax1.title.set_text('K562' )
    plt.imshow(HiC_K562[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 25)
    plt.grid()

    ax1 = plt.subplot(2, 3, 4)
    ax1.title.set_text('8x8 margin window' )
    plt.imshow( CNN_map_4[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower')
    plt.grid()
    plt.colorbar()

    ax1 = plt.subplot(2, 3, 5)
    ax1.title.set_text('8x8 window' )
    plt.imshow(CNN_map_8[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmax = 1)
    plt.grid()
    plt.colorbar()

    ax1 = plt.subplot(2, 3, 6)
    ax1.title.set_text('16x16 window' )
    plt.imshow(CNN_map_16[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmax = 1)
    plt.grid()
    plt.colorbar()

    #plt.show()
    plt.savefig('compare_window_size_addmargin_'+str(start), bbox_inches='tight')
    plt.close()

