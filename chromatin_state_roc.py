import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip

from scipy.stats import pearsonr
from scipy.stats import spearmanr

def d(M1, M2):
    return np.sqrt(np.sum((M1 - M2) ** 2))


path = '/home/zhangyan/triplet_loss'

chrN = 18
subImage_size = 100
chrN_start = chrN
chrN_end = chrN

'''
bin_diff = np.load('state_diff_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
bin_union = np.load('state_share_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
pos_index = []
neg_index = []
for i in range(subImage_size/2, bin_diff.shape[0]-subImage_size/2-1):
    if (bin_diff[i] == bin_union[i] and bin_union[i] >= 1000):
        pos_index.append(i-subImage_size/2)
        continue
    if (bin_diff[i] == 0 and bin_union[i] >= 1000):
        neg_index.append(i-subImage_size/2)

print "number of positive", len(pos_index)
print "number of negative", len(neg_index)
'''
pos_index = []
neg_index = []

epi_GM12878 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_Gm12878_chr18_epi_10kb.npy')[:,0]
epi_K562 = np.load('/home/zhangyan/predictHiC_new/Roadmap/divided_mean_K562_chr18_epi_10kb.npy')[:,0]
upper = 2
lower = 1
for i in range(subImage_size/2, epi_GM12878.shape[0]-subImage_size/2-1):
    if (epi_GM12878[i] >= 2 and epi_K562[i] < 1):
        pos_index.append(i-subImage_size/2)
    elif (epi_K562[i] >= 2 and epi_GM12878[i] < 1):
        pos_index.append(i-subImage_size/2)
    elif (epi_GM12878[i] >= 2 and epi_K562[i] >= 2):
        neg_index.append(i-subImage_size/2)


print "number of positive", len(pos_index)
print "number of negative", len(neg_index)


'''
entire_looplist = np.load('/home/zhangyan/looplist/GM12878_and_K562_loop.npy' ).item()
print entire_looplist
pos_index = []
neg_index = []


for key in entire_looplist.keys():
    loops = entire_looplist[key]
    if ((key[1] + key[2])/2 >= 7707):
        continue
    #if (1 in loops and 3 in loops):
    #    neg_index.append((key[1] + key[2])/2)
    if (1 in loops and (3 not in loops and not 2 in loops)):
        pos_index.append((key[1] + key[2])/2)        
    else:
        neg_index.append((key[1] + key[2])/2)

print "number of positive", len(pos_index)
print "number of negative", len(neg_index)
'''


anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)



prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")
pos_dis = np.zeros((output_pos_npy.shape[0]))
neg_dis = np.zeros((output_pos_npy.shape[0]))
K562_dis = np.zeros((output_pos_npy.shape[0]))
for i in range(0, output_anchor_npy.shape[0],1): 
    pos_dis[i] = d(output_anchor_npy[i], output_pos_npy[i])
    neg_dis[i] = d(output_anchor_npy[i], output_neg_npy[i])
    K562_dis[i] = d(output_anchor_npy[i], output_K562_npy[i])



print 'shape of our approach', output_pos_npy.shape, K562_dis.shape
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))

pixel_pos = np.zeros(HiC_anchor.shape[0])
pixel_neg = np.zeros(HiC_anchor.shape[0])
pixel_K562 = np.zeros(HiC_anchor.shape[0])
for i in range(0, HiC_anchor.shape[0] - subImage_size-1):
    pixel_pos[i] = np.sum(np.abs(positive[i][0] - anchor[i][0]))
    pixel_neg[i] = np.sum(np.abs(negative[i][0] - anchor[i][0]))
    pixel_K562[i] = np.sum(np.abs(K562[i][0] - anchor[i][0]))
print "pixel wise distance shape", pixel_pos.shape


Pearson_pos = np.load(path + '/pearson_pos_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_neg = np.load(path + '/pearson_neg_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_K562 = np.load(path + '/pearson_K562_vs_anchor_size100_chr'+str(chrN)+'.npy')
print 'Pearson', Pearson_pos.shape









#anchor_intensity = np.load(path + '/GM12878_primary_total_interaction_underdistance_' + str(distance) + '.npy')

#pos_intensity = np.load(path + '/GM12878_replicate_total_interaction_underdistance_' + str(distance) + '.npy')

#neg_intensity = np.load(path + '/IMR90_total_interaction_underdistance_' + str(distance) + '.npy')

#K562_intensity = np.load(path + '/K562_total_interaction_underdistance_' + str(distance) + '.npy')

K562_dis = K562_dis - pos_dis
pixel_K562 = pixel_K562- pixel_pos 
print "average distance for pos is ", np.mean(K562_dis[pos_index]), np.mean(K562_dis[neg_index])
print "average distance for pos is ", np.mean(pixel_K562[pos_index]), np.mean(pixel_K562[neg_index])
score0 = np.concatenate((K562_dis[pos_index], K562_dis[neg_index]))
score1 = np.concatenate((pixel_K562[pos_index], pixel_K562[neg_index]))
Pearson_score = np.concatenate((Pearson_K562[pos_index], Pearson_K562[neg_index]))


#Pearson_score = np.concatenate((K562_intensity[pos_index_2], K562_intensity[neg_index_2])) 
y = np.zeros((len(pos_index) + len(neg_index)))

y[:len(pos_index)] = 1
y_random = y.copy()
np.random.shuffle(y_random)
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

from scipy import interp

def genAUC(y_score, y_test):

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    #print "printing fpr"
    #print fpr
    #print "printing threshold"
    #print _
    return fpr, tpr, roc_auc

def Precision_Recall(y_score, y_test):
    precision, recall, _ = precision_recall_curve(y_test,y_score)
    average_precision = average_precision_score(y_test, y_score)
    return precision, recall, average_precision


plt.figure()
lw = 2
fpr1, tpr1, roc_auc1 = genAUC(score0, y)
plt.plot(fpr1, tpr1, color='black',
         lw=lw, label='Our approach (area = %0.4f)' % roc_auc1)

fpr2, tpr2, roc_auc2 = genAUC(score1, y)
plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='pixels wise curve (area = %0.4f)' % roc_auc2)

fpr3, tpr3, roc_auc3 = genAUC(Pearson_score, y)
plt.plot(fpr3, tpr3, color='blue',
         lw=lw, label='Pearson ROC curve (area = %0.4f)' % roc_auc3)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



