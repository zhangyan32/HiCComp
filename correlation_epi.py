import numpy as np
import model

from time import gmtime, strftime
import gzip
import matplotlib.pyplot as plt
import sys
import random
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def heatmap_to_1d(heatmap):
    result = np.zeros(heatmap.shape[0])
    for i in range(0, result.shape[0]):
        result[i] = np.nansum(heatmap[i,:]) + np.nansum(heatmap[:,i])
        #print result[i]
    return result

def genActiveIndex(array1d, threshold):
    pos = []
    for i in range(0, array1d.shape[0]):
        if (array1d[i] > threshold):
            pos.append(i)
    return pos


path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
subImage_size = 100

	

feature_map_pearson = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pearson_important_samples.npy') * -1
feature_map_pixel = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples.npy')
feature_map_dl = np.load(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples.npy')
'''
print "start making index"
index = [[],[]]
for i in range(0, feature_map_pearson.shape[0]):
    for j in range(i+1, feature_map_pearson.shape[0]):
    	if (j - i > 100 or j-i < 10):
    		index[0].append(j)
        	index[1].append(i)
        index[0].append(i)
        index[1].append(j)
print "finish making index"

feature_map_pearson[index] = 0
feature_map_pixel[index] = 0
feature_map_dl[index] = 0
print "finish change matrix"

np.save(path + '/K562vsGM12878_chr18_size100_step10_Pearson_important_samples_trim.npy', feature_map_pearson) 
np.save(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples_trim.npy', feature_map_pixel)
np.save(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples_trim.npy', feature_map_dl)
'''
print feature_map_dl.shape, feature_map_pixel.shape, feature_map_pearson.shape
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
#feature_important_map = np.load(path + '/K562_chr18_important_samples.npy')
subImage_size = 100

dl_1d = heatmap_to_1d(feature_map_dl)
pixel_1d = heatmap_to_1d(feature_map_pixel)
pearson_1d = heatmap_to_1d(feature_map_pearson)
#dl_1d[7000:] = 0
#pixel_1d[7000:] = 0
#pearson_1d[7000:] = 0
print pearsonr(dl_1d, pixel_1d)

plt.plot(range(0, dl_1d.shape[0]), dl_1d, color='r')
plt.plot(range(0, dl_1d.shape[0]), pixel_1d/(np.mean(pixel_1d)) * np.mean(dl_1d), color='b')
plt.show()


for i in range(0, 13):
    print "----------------------------------------", i
    epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,i]
    epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,i]
    epi_sum = epi_GM12878 + epi_K562
    epi_diff = np.abs(epi_GM12878 - epi_K562)/(epi_GM12878 + epi_K562 + 0.0001)
    #print np.percentile(epi_sum, 95), np.percentile(epi_sum, 90), np.percentile(epi_sum, 75), np.percentile(epi_sum, 50)

    index = genActiveIndex(epi_sum, np.percentile(epi_sum, 99))
    print pearsonr(dl_1d[index], epi_diff[index])[0], pearsonr(dl_1d[index], epi_sum[index])[0]
    print pearsonr(pixel_1d[index], epi_diff[index])[0], pearsonr(pixel_1d[index], epi_sum[index])[0]
    print pearsonr(pearson_1d[index], epi_diff[index])[0], pearsonr(pearson_1d[index], epi_sum[index])[0]

