import numpy as np
import model

from time import gmtime, strftime
import gzip
import matplotlib.pyplot as plt
import sys
import random
random.seed(321)

def ArrayAtDistance(matrix, distance):
    length = matrix.shape[0]
    index = [[], []]
    result = []
    for i in range(0, (length-distance)):
        index[0].append(i)
        index[1].append(i + distance)
        
    return matrix[index]

def divided_by_distance_mean(matrix, upper= 100):
    distanceMean = []
    for distance in range(0, upper):
        distanceMean.append(np.mean(ArrayAtDistance(matrix, distance)))
    result = np.zeros(matrix.shape)

    for i in range(0, matrix.shape[0]):
        for j in range(i, matrix.shape[1]):
            if (j - i >= upper):
                continue
            result[i][j] = matrix[i][j]/distanceMean[j-i]
    return result



path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
cell = 'K562'
#feature_important_map = np.load(path + '/K562_chr18_important_samples.npy')
subImage_size = 100
for latent_index in range(0, 1):
    #feature_important_map = np.load(path + '/K562vsGM12878_chr18_size100_step10_important_samples_0_0.npy')
    #feature_important_map = np.load(path + '/revise_K562vsGM12878_chr18_size100_step10_933_7l_important_samples_addmargin_0_8.npy')
    feature_important_map = np.load(path + '/revise_K562vsGM12878_chr18_size100_step10_933_7l_important_samples_addmargin_0_8_all_latent'+ '.npy')


    index = np.load(path + '/'+cell+'_diag_step1_index_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy')
    print feature_important_map.shape


    feature_map = np.zeros((7807, 7807))
    added_times = np.zeros((7807, 7807))
    for i in range(0, 7699, 10):
        #x = index[i][2]
        feature_map[i:i+subImage_size, i:i+subImage_size] += feature_important_map[i/10] 
        added_times[i:i+subImage_size, i:i+subImage_size] += 1
    feature_map = -feature_map / (added_times)


    np.save(path + '/revised_K562vsGM12878_chr18_size100_step10_CNN_important_samples_addmargin_windows_8_all_latent' , feature_map)
sys.exit()


HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
#HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
#HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_anchor_mean = divided_by_distance_mean(HiC_anchor)
HiC_K562_mean = divided_by_distance_mean(HiC_K562)
print "finish Generating distance mean"
pixel_different = np.abs(HiC_K562_mean - HiC_anchor_mean)







feature_map = pixel_different[0:7807, 0:7807] / np.minimum(1, added_times)

flatten_map = feature_map.flatten()
flatten_map = flatten_map[np.logical_not(np.logical_or(np.isnan(flatten_map), np.isinf(flatten_map)))]
feature_map[np.logical_or(np.isnan(feature_map), np.isinf(feature_map))] = -1
print flatten_map.shape
#sys.exit()
print np.percentile(flatten_map, 99.9)
print np.percentile(flatten_map, 99.5)
print np.percentile(flatten_map, 99)
print np.percentile(flatten_map, 98)
print np.percentile(flatten_map, 95)
print np.percentile(flatten_map, 90)
print np.percentile(flatten_map, 75)
print np.percentile(flatten_map, 50)
for i in range(0, 13):
    print "----------------------------------------", i
    epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,i]

    epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,i]


    #print epi_K562.shape, epi_GM12878.shape
    #print epi_GM12878, epi_K562
    #epi_K562 = np.minimum(5, epi_K562)
    #epi_GM12878 = np.minimum(5, epi_GM12878)
    epi_distance = np.abs((epi_K562 - epi_GM12878))/(epi_K562 + epi_GM12878+0.0001)
    #feature_map = np.maximum(1.2, feature_map)

    key_index = np.where(feature_map > np.percentile(flatten_map, 99.99))
    index = list(set(key_index[0]) | set(key_index[1]))


    neg_index = list(set(range(0, epi_distance.shape[0])) - set(index))

    print len(index), len(neg_index)
    #print index
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




    print "none empty", np.trim_zeros(epi_K562[neg_index] + epi_GM12878[neg_index]).shape
    print "no resample", np.mean(np.trim_zeros(epi_distance[neg_index])), np.mean(np.trim_zeros(epi_K562[neg_index] + epi_GM12878[neg_index]))
    print "resample", np.mean(np.trim_zeros(epi_distance[neg_index_sample])), np.mean(np.trim_zeros(epi_K562[neg_index_sample] + epi_GM12878[neg_index_sample]))
    print  "detection", np.mean(np.trim_zeros(epi_distance[index])), np.mean(np.trim_zeros(epi_K562[index] + epi_GM12878[index]))


    from scipy import stats


    print stats.ttest_ind(list(np.trim_zeros(epi_distance[index])), list(np.trim_zeros(epi_distance[neg_index_sample])))
    print stats.ttest_ind(list(np.trim_zeros(epi_sum[index])), list(np.trim_zeros(epi_sum[neg_index_sample])))
    print stats.ttest_ind(list(np.trim_zeros(epi_distance[index])), list(np.trim_zeros(epi_distance[neg_index])))

sys.exit()
plt.hist(np.trim_zeros(epi_K562[neg_index] + epi_GM12878[neg_index]), bins = 30)
plt.show()
plt.close()
plt.hist(np.trim_zeros(epi_K562[index] + epi_GM12878[index]), bins = 30)
plt.show()

for start in range(0, 2000, 180):
    print start
    end = start + 200
    plt.figure(figsize=(30, 30))
    ax1 = plt.subplot(2, 3, 1)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(2, 3, 2)
    ax1.title.set_text('GM12878' )
    plt.imshow(HiC_pos[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(2, 3, 3)
    ax1.title.set_text('K562' )
    plt.imshow(HiC_K562[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(2, 3, 4)
    ax1.title.set_text('importance' )
    plt.imshow( feature_map[start:end, start:end], cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 2)
    plt.grid()
    plt.colorbar()

    ax1 = plt.subplot(2, 3, 5)
    ax1.title.set_text('pixels difference' )
    plt.imshow(np.abs(HiC_K562[start:end, start:end] - HiC_anchor[start:end, start:end]), cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 50)
    plt.grid()
    plt.colorbar()

    ax1 = plt.subplot(2, 3, 6)
    ax1.title.set_text('pixels same' )
    plt.imshow(np.abs(HiC_pos[start:end, start:end] - HiC_anchor[start:end, start:end]), cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 50)
    plt.grid()
    plt.colorbar()

    plt.show()
    #plt.savefig('', bbox_inches='tight')
    plt.close()