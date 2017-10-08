import numpy as np
import model

from time import gmtime, strftime
import gzip
import matplotlib.pyplot as plt
import sys
import random
from scipy import stats

def heatmap_to_1d(heatmap):
    result = np.zeros(heatmap.shape[0])
    for i in range(0, result.shape[0]):
        result[i] = np.nansum(heatmap[i,:]) + np.nansum(heatmap[:,i])
        #print result[i]

    return result

def genIndex(array1d, threshold):
    pos = []
    neg = []
    for i in range(0, array1d.shape[0]):
        if (array1d[i] > threshold):
            pos.append(i)
        else:
            neg.append(i)
    return pos, neg

def genSignificant(array1d, c1, c2, caller):
    pos, neg = genIndex(array1d, np.percentile(array1d, 99.5))
    percentage = np.abs(c1 - c2)/(c1 + c2 + 0.001)
    _sum = c1 + c2
    group1 = percentage[pos]
    group2 = percentage[neg]
    print caller, len(pos), len(neg), np.mean(_sum[pos]), np.mean(_sum[neg])
    return stats.ttest_ind(list(np.trim_zeros(group1)), list(np.trim_zeros(group2))), np.mean(group1), np.mean(group2)


def genSignificant_resample(array1d, c1, c2, caller):
    pos, neg = genIndex(array1d, np.percentile(array1d, 90))
    percentage = np.abs(c1 - c2)/(c1 + c2 + 0.001)
    epi_sum = c1 + c2

    neg_distr = np.histogram(np.trim_zeros(epi_sum[neg]), bins=10, range=(0, 10))[0]
    pos_distr = np.histogram(np.trim_zeros(epi_sum[pos]), bins=10, range=(0, 10))[0]
    ratio = pos_distr / (1.0 * neg_distr)
    neg_index_sample = []
    for i in neg:
        group = min(9, int(epi_sum[i]))
        #print group, ratio[group]
        if (random.random() < ratio[group]):
            neg_index_sample.append(i)
    group1 = percentage[pos]
    group2 = percentage[neg_index_sample]
    print caller, len(pos), len(neg_index_sample), np.mean(epi_sum[pos]), np.mean(epi_sum[neg_index_sample])
    return stats.ttest_ind(list(np.trim_zeros(group1)), list(np.trim_zeros(group2))), np.mean(group1), np.mean(group2)






    epi_sum = epi_K562 + epi_GM12878
    neg_dis = np.histogram(np.trim_zeros(epi_sum[neg_index]), bins=10, range=(0, 10))[0]
    pos_dis = np.histogram(np.trim_zeros(epi_K562[index] + epi_GM12878[index]), bins=10, range=(0, 10))[0]
    ratio = pos_dis / (1.0 * neg_dis)
    neg_index_sample = []
    for i in neg_index:
        group = min(9, int(epi_sum[i]))
        #print group, ratio[group]
        if (random.random() < ratio[group]):
            neg_index_sample.append(i)



path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
subImage_size = 100


feature_map_pearson = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pearson_important_samples.npy') * -1
feature_map_pixel = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples.npy')
feature_map_dl = np.load(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples.npy')

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
dl_1d[7000:] = 0
pixel_1d[7000:] = 0
pearson_1d[7000:] = 0
for i in range(0, 4):
    print "----------------------------------------", i
    epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,i]
    epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,i]
    print "CNN", genSignificant_resample(dl_1d, epi_GM12878, epi_K562, "CNN")
    print "pixel",genSignificant_resample(pixel_1d, epi_GM12878, epi_K562, "Pixel")
    print "pearson",genSignificant_resample(pearson_1d, epi_GM12878, epi_K562, "Pearson")



