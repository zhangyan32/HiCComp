import numpy as np
import model

from time import gmtime, strftime
import gzip
import matplotlib.pyplot as plt
import sys
import random
from scipy import stats
def ArrayAtDistance(matrix, distance):
    length = matrix.shape[0]
    index = [[], []]
    result = []
    for i in range(0, (length-distance)):
        index[0].append(i + distance)
        index[1].append(i)
        
    return matrix[index]

def divided_by_distance_mean(matrix, upper= 100):
    distanceMean = []
    for distance in range(0, upper):
        distanceMean.append(np.nanmean(ArrayAtDistance(matrix, distance)))
    print distanceMean
    result = np.zeros(matrix.shape)

    for i in range(0, matrix.shape[0]):
        for j in range(i, matrix.shape[1]):

            if (j - i >= upper):
                continue
            #print j, i, matrix[j][i], distanceMean[j-i]
            result[j][i] = matrix[j][i]/distanceMean[j-i]
 
    return result
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

    pos, neg = genIndex(array1d, np.percentile(array1d, 95))
    percentage = np.abs(c1 - c2)/(c1 + c2 + 0.001)
    _sum = c1 + c2
    group1 = percentage[pos]
    group2 = percentage[neg]
    print caller, "pos:", len(pos), "neg:", len(neg)
    return stats.ttest_ind(list(np.trim_zeros(group1)), list(np.trim_zeros(group2))), np.mean(group1), np.mean(group2), np.mean(_sum[pos]), np.mean(_sum[neg])

def gen2dSignificant_resample(heatmap, c1, c2, p = 99.9,caller = ''):
    flatten_map = heatmap.flatten()
    flatten_map = flatten_map[np.logical_not(np.logical_or(np.isnan(flatten_map), np.isinf(flatten_map)))]
    theshold = np.percentile(flatten_map,p )
    key_index = np.where(heatmap > theshold)
    #distance = key_index[1] - key_index[0]
    #print caller, "75%", np.percentile(distance, 75), "50%", np.percentile(distance, 50), "25%", np.percentile(distance, 25)
    #pos_index = list(set(key_index[0]) | set(key_index[1]))
    pos_index = []
    pos_index.extend(key_index[0])
    pos_index.extend(key_index[1])
    print len(pos_index)
    neg_index = list(set(range(0, c1.shape[0])) - set(pos_index))
    print caller, "pos:", len(pos_index), "neg:", len(neg_index)
    percentage = np.abs(c1 - c2)/(c1 + c2 + 0.001)
    epi_sum = c1 + c2
    neg_distr = np.histogram(np.trim_zeros(epi_sum[neg_index]), bins=20, range=(0, 20))[0]
    pos_distr = np.histogram(np.trim_zeros(epi_sum[pos_index]), bins=20, range=(0, 20))[0]
    ratio = pos_distr / (1.0 * neg_distr)
    neg_index_sample = []
    while(True):
        random.shuffle(neg_index)
        for i in neg_index:
            group = min(19, int(epi_sum[i]+0.5))
            #print group, ratio[group]
            if (random.random() < ratio[group]):
                neg_index_sample.append(i)
                if (len(neg_index_sample) == len(pos_index)):
                    return pos_index, neg_index_sample

def gen2dSignificant(heatmap, c1, c2, p = 99.9, caller = ''):
    flatten_map = heatmap.flatten()
    flatten_map = flatten_map[np.logical_not(np.logical_or(np.isnan(flatten_map), np.isinf(flatten_map)))]
    #print p
    theshold = np.percentile(flatten_map, p)
    key_index = np.where(heatmap > theshold)
    print len(key_index[0])
    distance = key_index[1] - key_index[0]
    #print caller, "75%", np.percentile(distance, 75), "50%", np.percentile(distance, 50), "25%", np.percentile(distance, 25)
    #pos_index = list(set(key_index[0]) | set(key_index[1]))
    pos_index = []
    pos_index.extend(key_index[0])
    pos_index.extend(key_index[1])
    neg_index = list(set(range(0, c1.shape[0])))
    #neg_index = list(set(range(0, c1.shape[0])) - set(pos_index))
    try:
        print caller, "pos:", len(pos_index), "neg:", len(neg_index)
    except:
        pass
    percentage = np.abs(c1 - c2)/(c1 + c2 + 0.001)
    _sum = c1 + c2
    group1 = percentage[pos_index]
    group2 = percentage[neg_index]
    return pos_index, neg_index    
    #return stats.ttest_ind(list(np.trim_zeros(group1)), list(np.trim_zeros(group2))), np.mean(group1), np.mean(group2), np.mean(_sum[pos_index]), np.mean(_sum[neg_index])

def t_test(pos_index, neg_index, array1d):
    group1 = array1d[pos_index]
    group2 = array1d[neg_index]   
    return stats.ttest_ind(list(np.trim_zeros(group1)), list(np.trim_zeros(group2))) 

path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
subImage_size = 100


feature_map_pearson = np.load(path + '//revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_all_latent.npy') 
feature_map_pixel = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_all_latent.npy')
feature_map_dl = np.load(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_0.npy')
#feature_map_pixel = np.load(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples_nan_distance_mean.npy')
#feature_map_dl = np.load(gzip.GzipFile(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples_nan.npy.gz', "r"))
#feature_map_dl = np.load(gzip.GzipFile(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples_nan.npy.gz', "r"))
'''
feature_map_pixel = divided_by_distance_mean(feature_map_pixel)

index = [[],[]]
for i in range(0, feature_map_dl.shape[0]):
    for j in range(i+1, feature_map_dl.shape[0]):
        if (j - i > 100):
            index[0].append(j)
            index[1].append(i)
        index[0].append(i)
        index[1].append(j)
feature_map_pearson[index] = float('nan')
feature_map_pixel[index] = float('nan')
feature_map_dl[index] = float('nan')

start = 0
end = 500

plt.imshow(feature_map_pixel[start:end, start:end],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmax=1)
plt.axis('off')
plt.colorbar()
plt.show()

np.save(path + '/K562vsGM12878_chr18_size100_step10_Pearson_important_samples_nan.npy', feature_map_pearson) 
np.save(path + '/K562vsGM12878_chr18_size100_step10_Pixel_important_samples_nan_distance_mean.npy', feature_map_pixel)
np.save(path + '/K562vsGM12878_chr18_size100_step10_CNN_important_samples_nan.npy', feature_map_dl)
'''
'''

for start in range(10000, 7000, 150):
    print start
    end = start + 200
    plt.figure(figsize=(30, 30))
    ax1 = plt.subplot(1, 3, 1)
    ax1.title.set_text('CNN' )
    plt.imshow(feature_map_dl[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower',)
    plt.colorbar()
    plt.grid()

    ax1 = plt.subplot(1, 3, 2)
    ax1.title.set_text('Pearson' )
    plt.imshow(feature_map_pearson[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower', )
    plt.colorbar()
    plt.grid()

    ax1 = plt.subplot(1, 3, 3)
    ax1.title.set_text('pixel' )
    plt.imshow(feature_map_pixel[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower', )
    plt.grid()

    plt.colorbar()

    #plt.show()
    plt.savefig('compare_important_map' + str(start) , bbox_inches='tight')
    plt.close()
'''
print feature_map_dl.shape, feature_map_pixel.shape, feature_map_pearson.shape
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
#feature_important_map = np.load(path + '/K562_chr18_important_samples.npy')
subImage_size = 100

#outfile = open('chip_seq.csv', 'r')
result = []
for i in range(0, 13):
    print "----------------------------------------", i
    epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,i]
    epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,i]
    epi_diff = np.abs(epi_GM12878 - epi_K562)/np.abs(epi_GM12878 + epi_K562 + 0.001)
    epi_sum = np.abs(epi_GM12878 + epi_K562)
    top = 1000.0
    penc_dl = 1 - top/np.count_nonzero(~np.isnan(feature_map_dl))
    penc_pixel = 1 - top/np.count_nonzero(~np.isnan(feature_map_pixel))

    penc_pearson = 1 - top/np.count_nonzero(~np.isnan(feature_map_pearson))
    print "percentage", penc_dl, penc_pixel
    p_cnn, n_cnn = gen2dSignificant(feature_map_dl, epi_GM12878, epi_K562, p =penc_dl  * 100,caller='CNN')
    p_pixel, n_pixel = gen2dSignificant(feature_map_pixel, epi_GM12878,  epi_K562, p = penc_pixel * 100,caller='pixel')
    p_pearson, n_pearson = gen2dSignificant(feature_map_pearson, epi_GM12878, epi_K562, p = penc_pearson * 100,caller='pearson')
    Np = min(len(p_cnn), min(len(p_pixel), len(p_pearson)))
    Nn = min(len(n_cnn), min(len(n_pixel), len(n_pearson)))
    random.shuffle(p_cnn)
    random.shuffle(n_cnn)
    random.shuffle(p_pixel)
    random.shuffle(n_pixel)
    random.shuffle(p_pearson)
    random.shuffle(n_pearson)
    #p_cnn = p_cnn[:Np]
    #n_cnn = n_cnn[:Nn]
    #p_pixel = p_pixel[:Np]
    #n_pixel = n_pixel[:Nn]
    #p_pearson = p_pearson[:Np]
    #n_pearson = n_pearson[:Nn]
    print len(p_cnn), len(p_pixel), len(p_pearson)
    print len(n_cnn), len(n_pixel), len(n_pearson)
    one_seq = [epi_diff[p_cnn], epi_diff[p_pixel], epi_diff[p_pearson], epi_diff[n_cnn], epi_diff[n_pixel], epi_diff[n_pearson]]
    result.append(one_seq)

    print t_test(p_cnn, n_cnn, epi_sum), np.mean(epi_diff[p_cnn]), np.mean(epi_diff[n_cnn]), np.mean(epi_sum[p_cnn]), np.mean(epi_sum[n_cnn])
    print t_test(p_pixel, n_pixel, epi_sum), np.mean(epi_diff[p_pixel]), np.mean(epi_diff[n_pixel]), np.mean(epi_sum[p_pixel]), np.mean(epi_sum[n_pixel])
    print t_test(p_pearson, n_pearson, epi_sum), np.mean(epi_diff[p_pearson]), np.mean(epi_diff[n_pearson]), np.mean(epi_sum[p_pearson]), np.mean(epi_sum[n_pearson])

np.save('chip_seq_enrich_revised_feature_all_latent.npy', np.array(result))
