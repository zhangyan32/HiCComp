import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip

from scipy.stats import pearsonr
from scipy.stats import spearmanr

path = '/home/zhangyan/triplet_loss'

chrN = 18
subImage_size = 100
chrN_start = chrN
chrN_end = chrN
anchor = np.load(path + '/GM12878_primary_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicate_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
negative = np.load(path + '/IMR90_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
K562 = np.load(path + '/K562_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
print "shape of the samples", anchor.shape
batch_size = anchor.shape[0]/10 + 1

def d(M1, M2):
    return np.sqrt(np.sum((M1 - M2) ** 2))
    #return (np.sum((M1 - M2) ** 2))
#prefix = 'chr' + str(chrN_end) + '_test_'
if (0):
    prefix = 'chr'+str(chrN_start)+'_test_size100_'

    output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1_maxMargin.npy")
    output_pos_npy = np.load(path + '/' + prefix + "pos_step1_maxMargin.npy")
    output_neg_npy = np.load(path + '/' + prefix + "neg_step1_maxMargin.npy")
    output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1_maxMargin.npy")
else:
    prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_'
    prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
    prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
    prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'
    output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1.npy")
    output_pos_npy = np.load(path + '/' + prefix + "pos_step1.npy")
    output_neg_npy = np.load(path + '/' + prefix + "neg_step1.npy")
    output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1.npy")
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))

print positive.shape

pixel_pos = np.zeros(HiC_anchor.shape[0])
pixel_neg = np.zeros(HiC_anchor.shape[0])
pixel_K562 = np.zeros(HiC_anchor.shape[0])
for i in range(subImage_size/2, HiC_anchor.shape[0] - subImage_size/2):
    pixel_pos[i] = np.sum(np.abs(positive[i-subImage_size/2][0] - anchor[i-subImage_size/2][0]))
    pixel_neg[i] = np.sum(np.abs(negative[i-subImage_size/2][0] - anchor[i-subImage_size/2][0]))
    pixel_K562[i] = np.sum(np.abs(K562[i-subImage_size/2][0] - anchor[i-subImage_size/2][0]))
print "pixel wise distance shape", pixel_pos.shape
count = 0
for i in range(0, pixel_pos.shape[0]):
    if (pixel_pos[i] - pixel_K562[i] > 0):
        count += 1
#pixel_K562 /= np.mean(pixel_K562)
#pixel_K562 /= (pixel_pos + 0.001)
print "number of error is ", count
print "pixel wise distance shape", pixel_pos.shape



'''
epi_GM12878 = np.load('state_10kb_Gm12878.npy')[17][:,5]
epi_K562 = np.load('state_10kb_K562.npy')[17][:,5]
print epi_K562.shape
for i in range(0, 1):
    epi_GM12878[i] /= np.mean(epi_GM12878)
    epi_K562[i] /= np.mean(epi_K562)
epi_distance = (epi_K562 - epi_GM12878) * (epi_K562 - epi_GM12878)
epi_overall = (epi_K562 + epi_GM12878) * (epi_K562 + epi_GM12878)

#epi_distance = np.sum(epi_distance, axis=1)
#epi_overall = np.sum(epi_overall, axis=1)
print "epi distance shape", epi_overall.shape
epi_ratio = epi_distance / (epi_overall + 0.01)
epi_ratio = epi_distance 
epi_ratio_smooth = np.zeros(epi_ratio.shape)
print "large value of non smooth epi", np.amax(epi_ratio)



print "epi marker curve shape", epi_ratio.shape
'''

Pearson_pos = np.load(path + '/pearson_pos_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_neg = np.load(path + '/pearson_neg_vs_anchor_size100_chr'+str(chrN)+'.npy')
Pearson_K562 = np.load(path + '/pearson_K562_vs_anchor_size100_chr'+str(chrN)+'.npy')



bin_diff = np.load('state_diff_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')
bin_union = np.load('state_share_K562_GM12878_CTCF_chr' + str(chrN) + '.npy')



pos_index = []
neg_index = []
for i in range(subImage_size/2, bin_diff.shape[0]-subImage_size/2):
    if (bin_diff[i] == bin_union[i] and bin_union[i] != 0):
        pos_index.append(i)
    if (bin_diff[i] == 0):
        neg_index.append(i)

CTCF_state = np.zeros(bin_diff.shape)
CTCF_state[pos_index] = 2
CTCF_state[neg_index] = 1


print "number of positive", len(pos_index), "; number of negative", len(neg_index)
pos_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
neg_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
K562_dis = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
for i in range(subImage_size/2, output_anchor_npy.shape[0],1): 
    pos_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_pos_npy[i-subImage_size/2])
    neg_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_neg_npy[i-subImage_size/2])
    K562_dis[i] = d(output_anchor_npy[i-subImage_size/2], output_K562_npy[i-subImage_size/2])

kernel = np.zeros(subImage_size)
for i in range(0, subImage_size/2):
    kernel[i] = i
    kernel[subImage_size-1-i] = i
print kernel[23:27]
kernel /= np.sum(kernel)

pos_dis_smooth = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
neg_dis_smooth = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
K562_dis_smooth = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
epi_ratio_smooth = np.zeros((output_pos_npy.shape[0]+subImage_size+1,))
for i in range(subImage_size/2, output_pos_npy.shape[0] - subImage_size/2):
    pos_dis_smooth[i] = np.sum(kernel * pos_dis[i-subImage_size/2:i+subImage_size/2]) * 5
    neg_dis_smooth[i] = np.sum(kernel * neg_dis[i-subImage_size/2:i+subImage_size/2]) * 5
    K562_dis_smooth[i] = np.sum(kernel * K562_dis[i-subImage_size/2:i+subImage_size/2]) * 5
    epi_ratio_smooth[i] = np.sum(bin_diff[i-subImage_size/2:i+subImage_size/2] * kernel)
print "mean of pos ", np.nanmean(pos_dis_smooth), np.mean(pos_dis_smooth)
print "mean of neg before ", np.nanmean(neg_dis_smooth), np.mean(neg_dis_smooth)
print "mean of K562 before ", np.nanmean(K562_dis_smooth), np.mean(K562_dis_smooth)
#K562_dis_smooth = K562_dis_smooth / pos_dis_smooth
#neg_dis_smooth = neg_dis_smooth / pos_dis_smooth
#K562_dis -= (pos_dis + 0.01)
#neg_dis -= (pos_dis + 0.01)
K562_loss = pos_dis - K562_dis + 1
neg_loss = pos_dis - neg_dis + 1
pixel_K562 /= (pixel_pos+ 0.01)
pixel_neg /= (pixel_pos+ 0.01)
pixel_pos /= np.mean(pixel_pos)


#K562_dis /= (pos_dis + 0.01)
K562_dis_smooth /= (pos_dis_smooth + 0.01)
neg_dis_smooth /= (pos_dis_smooth + 0.01)
pos_dis_smooth /= np.nanmean(pos_dis_smooth)
K562_dis_smooth /= np.nanmean(K562_dis_smooth)
neg_dis_smooth /= np.nanmean(neg_dis_smooth)
print "mean of neg after ", np.nanmean(neg_dis_smooth), np.mean(neg_dis_smooth)
print "mean of K562 after ", np.nanmean(K562_dis_smooth), np.mean(K562_dis_smooth)
print "large value of epi", np.amax(epi_ratio_smooth)
epi_ratio_smooth /= np.amax(epi_ratio_smooth)
epi_ratio_smooth *=6

distance = subImage_size
anchor_intensity = np.load(path + '/GM12878_primary_total_interaction_underdistance_' + str(distance) + '.npy')

pos_intensity = np.load(path + '/GM12878_replicate_total_interaction_underdistance_' + str(distance) + '.npy')

neg_intensity = np.load(path + '/IMR90_total_interaction_underdistance_' + str(distance) + '.npy')

K562_intensity = np.load(path + '/K562_total_interaction_underdistance_' + str(distance) + '.npy')


#neg_dis_smooth /= (pos_dis_smooth + 0.01)
#K562_dis_smooth /= (pos_dis_smooth + 0.01)
tobezero = [[],[]]
for i in range(0, HiC_pos.shape[0]):
    for j in range(i + subImage_size, HiC_pos.shape[0]):
        tobezero[0].append(j)
        tobezero[1].append(i)
HiC_anchor[tobezero] = 0
HiC_pos[tobezero] = 0
HiC_neg[tobezero] = 0
HiC_K562[tobezero] = 0

start= 1000
#K562_dis_smooth = K562_dis_smooth /(K562_intensity[0:7807] + 0.001)
#print pearsonr(K562_dis_smooth, epi_ratio_smooth[0:7807])[0]#, pearsonr(K562_dis_smooth, epi_ratio_smooth[1:7808])[0]
#print pearsonr(K562_intensity[0:7807], epi_ratio_smooth)
#rint pearsonr(K562_intensity[0:7807], K562_dis_smooth)

pos_index_2 = [x - subImage_size/2 for x in pos_index]
neg_index_2 = [x - subImage_size/2 for x in neg_index]

pos_index_3 = [x + subImage_size/2 for x in pos_index]
neg_index_3 = [x + subImage_size/2 for x in neg_index]

print K562_dis.shape

print "average distance for pos is ", np.mean(K562_dis_smooth[pos_index_3]), np.mean(K562_dis_smooth[neg_index_3])
print "average distance for pos is ", np.mean(pixel_K562[pos_index_2]), np.mean(pixel_K562[neg_index_2])
score = np.concatenate((K562_dis[pos_index], K562_dis_smooth[neg_index]))
score_2 = np.concatenate((pixel_K562[pos_index], pixel_K562[neg_index]))
score_3 = np.concatenate((K562_loss[pos_index]*-1, K562_loss[neg_index]*-1))
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

f, t, s = genAUC(score, y)
plt.figure()
lw = 2
fpr1, tpr1, roc_auc1 = genAUC(score_2, y)
plt.plot(fpr1, tpr1, color='black',
         lw=lw, label='Score2 curve (area = %0.4f)' % roc_auc1)
fpr2, tpr2, roc_auc2 = genAUC(score_3, y)
plt.plot(fpr2, tpr2, color='red',
         lw=lw, label='Score3 curve (area = %0.4f)' % roc_auc2)
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





#sys.exit()
epi_ratio_smooth = (epi_ratio_smooth)/np.median(epi_ratio_smooth)
chipseq_K562 = np.loadtxt("/home/zhangyan/epigenomic_features_10k/K562/10kb/chr18.epifeature")
chipseq_GM12878 = np.loadtxt("/home/zhangyan/epigenomic_features_10k/Gm12878/10kb/chr18.epifeature")
print chipseq_K562.shape, chipseq_GM12878.shape
chipseq_diff = np.sum(np.abs(chipseq_K562 - chipseq_GM12878) / (chipseq_K562 + chipseq_GM12878 + 0.001), axis = 1)
chipseq_diff /= np.mean(chipseq_diff)





for start in range(0, 7000, 200):
    end = start + 200
    plt.figure(figsize=(30, 30))
    ax1 = plt.subplot(3, 3, 1)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(3, 3, 2)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(3, 3, 3)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(3, 3, 4)
    ax1.title.set_text('GM12878' )
    plt.imshow(HiC_pos[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(3, 3, 5)
    ax1.title.set_text('IMR90' )
    plt.imshow(HiC_neg[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()

    ax1 = plt.subplot(3, 3, 6)
    ax1.title.set_text('K562' )
    plt.imshow(HiC_K562[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 50)
    plt.grid()


    x = range(start, end)
    ax1 = plt.subplot(3, 3, 7)
    
    plt.plot(x, pos_dis_smooth[start:end], color='k', label='vs Pos',linewidth=3)
    plt.plot(x, anchor_intensity[start:end], color='r', label='Anchor intensity',linewidth=3)
    plt.plot(x, pos_dis[start:end], color='b', label='istance',linewidth=3)
    plt.plot(x, pixel_pos[start:end] * 40, color='y', label='pixel',linewidth=3)
    #plt.plot(x, neg_dis[start:end], color='m', label='vs Neg',linewidth=3)    
    #plt.plot(x, K562_dis[start:end], color='b', label='vs K562',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    #plt.ylim(0, 6)
    plt.xlim(start, end)
    
    ax1 = plt.subplot(3, 3, 8)
    #plt.plot(x, pos_dis[start:end], color='k', label='vs Pos',linewidth=3)
    plt.plot(x, neg_dis_smooth[start:end], color='m', label='vs Neg',linewidth=3)  
    plt.plot(x, neg_loss[start:end], color='r', label='intensity',linewidth=5)  
    plt.plot(x, Pearson_neg[start:end]* 40, color='y', label='Pearson',linewidth=3)
    plt.plot(x, pixel_neg[start:end]* 40, color='k', label='pixel',linewidth=3)
    plt.plot(x, neg_dis[start:end], color='b', label='istance',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    #plt.ylim(0, 6)
    plt.xlim(start, end)


    ax1 = plt.subplot(3, 3, 9)
    #plt.plot(x, pos_dis[start:end], color='k', label='vs Pos',linewidth=3)
    #plt.plot(x, neg_dis[start:end], color='m', label='vs Neg',linewidth=3)    
    plt.plot(x, K562_dis_smooth[start:end], color='b', label='vs K562',linewidth=3)
    plt.plot(x, K562_loss[start:end], color='r', label='intensity',linewidth=5)
    plt.plot(x, pixel_K562[start:end]* 40, color='k', label='pixel',linewidth=3)
    plt.plot(x, Pearson_K562[start:end]* 40, color='y', label='Pearson',linewidth=3)
    plt.plot(x, K562_dis[start:end], color='b', label='istance',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    #plt.ylim(0, 6)
    plt.xlim(start, end)

    #plt.show()
    plt.savefig('weighted_size100_smooth_tripletloss_ratiotobaseline_compareheatmap_compare_' + str(start) + '_' + str(end), bbox_inches='tight')
    plt.close()
sys.exit()

'''
pos_dis = np.zeros((output_pos_npy.shape[0],))
neg_dis = np.zeros((output_pos_npy.shape[0],))
K562_dis = np.zeros((output_pos_npy.shape[0],))
for i in range(0, output_anchor_npy.shape[0],1): 
    pos_dis[i:i+50] += d(output_anchor_npy[i], output_pos_npy[i])
    neg_dis[i:i+50] += d(output_anchor_npy[i], output_neg_npy[i])
    K562_dis[i:i+50] += d(output_anchor_npy[i], output_K562_npy[i])

pos_dis /= 50
neg_dis /= 50
K562_dis /= 50

x = range(output_pos_npy.shape[0])
plt.plot(x, pos_dis, color='k', label='vs Pos',linewidth=3)
plt.plot(x, neg_dis, color='m', label='vs Neg',linewidth=3)    
plt.plot(x, K562_dis, color='b', label='vs K562',linewidth=3)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
#plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
#plt.ylim(0, 1)
plt.xlim(1000,2000)
plt.show()




sys.exit()
'''


for i in range(1000, output_anchor_npy.shape[0],50):
    print "-----------"
    print i
    print output_anchor_npy[i]
    print output_pos_npy[i]
    print output_neg_npy[i]
    print output_K562_npy[i]
    print "#################"
    ax1 = plt.subplot(2, 2, 1)
    ax1.title.set_text("original")
    plt.imshow(anchor[i][0],  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)



    ax1 = plt.subplot(2, 2, 2)
    ax1.title.set_text(str(d(output_anchor_npy[i], output_pos_npy[i]))  )
    plt.imshow(positive[i][0],  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  

    ax1 = plt.subplot(2, 2, 3)
    ax1.title.set_text(str(d(output_anchor_npy[i], output_neg_npy[i]))  )
    plt.imshow(negative[i][0],  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  

    ax1 = plt.subplot(2, 2, 4)
    ax1.title.set_text(str(d(output_anchor_npy[i], output_K562_npy[i])) + ", " + str(d(output_neg_npy[i], output_K562_npy[i]))  )
    plt.imshow(K562[i][0],  cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)
    plt.show()
    plt.close() 



