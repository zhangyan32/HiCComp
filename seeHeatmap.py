import numpy as np
from time import gmtime, strftime
use_gpu = 0
import sys
import matplotlib.pyplot as plt
import gzip
path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18
chrN = 18
anchor = np.load(path + '/GM12878_primary_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicate_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
negative = np.load(path + '/IMR90_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
K562 = np.load(path + '/K562_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
print "shape of the samples", anchor.shape
batch_size = anchor.shape[0]/10 + 1

def d(M1, M2):
    return np.sqrt(np.sum((M1 - M2) ** 2))


prefix = "chr18_test_"
output_anchor_npy = np.load(path + '/' + prefix + "anchor_step1_maxMargin.npy")
output_pos_npy = np.load(path + '/' + prefix + "pos_step1_maxMargin.npy")
output_neg_npy = np.load(path + '/' + prefix + "neg_step1_maxMargin.npy")
output_K562_npy = np.load(path + '/' + prefix + "output_K562_step1_maxMargin.npy")
HiC_anchor = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_primaryMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_pos = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/GM12878_replicateMAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_neg = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/IMR90MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))
HiC_K562 = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy.gz', "r"))


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
tobezero = [[],[]]
for i in range(0, HiC_pos.shape[0]):
    for j in range(i + 50, HiC_pos.shape[0]):
        tobezero[0].append(i)
        tobezero[1].append(j)
HiC_anchor[tobezero] = 0
HiC_pos[tobezero] = 0
HiC_neg[tobezero] = 0
HiC_K562[tobezero] = 0

start= 1000
for start in range(0, 7000, 200):
    end = start + 200
    plt.figure(figsize=(30, 30))
    ax1 = plt.subplot(3, 3, 1)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 99)
    plt.grid()

    ax1 = plt.subplot(3, 3, 2)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 99)
    plt.grid()

    ax1 = plt.subplot(3, 3, 3)
    ax1.title.set_text('GM12878_primary' )
    plt.imshow(HiC_anchor[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 99)
    plt.grid()

    ax1 = plt.subplot(3, 3, 4)
    ax1.title.set_text('GM12878' )
    plt.imshow(HiC_pos[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 88)
    plt.grid()

    ax1 = plt.subplot(3, 3, 5)
    ax1.title.set_text('IMR90' )
    plt.imshow(HiC_neg[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 44)
    plt.grid()

    ax1 = plt.subplot(3, 3, 6)
    ax1.title.set_text('K562' )
    plt.imshow(HiC_K562[start:end, start:end], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 33)
    plt.grid()


    x = range(start, end)
    ax1 = plt.subplot(3, 3, 7)
    plt.plot(x, pos_dis[start:end], color='k', label='vs Pos',linewidth=3)
    #plt.plot(x, neg_dis[start:end], color='m', label='vs Neg',linewidth=3)    
    #plt.plot(x, K562_dis[start:end], color='b', label='vs K562',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    plt.ylim(0, 3)
    plt.xlim(start, end)
    
    ax1 = plt.subplot(3, 3, 8)
    #plt.plot(x, pos_dis[start:end], color='k', label='vs Pos',linewidth=3)
    plt.plot(x, neg_dis[start:end], color='m', label='vs Neg',linewidth=3)    
    #plt.plot(x, K562_dis[start:end], color='b', label='vs K562',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    plt.ylim(0, 3)
    plt.xlim(start, end)


    ax1 = plt.subplot(3, 3, 9)
    #plt.plot(x, pos_dis[start:end], color='k', label='vs Pos',linewidth=3)
    #plt.plot(x, neg_dis[start:end], color='m', label='vs Neg',linewidth=3)    
    plt.plot(x, K562_dis[start:end], color='b', label='vs K562',linewidth=3)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.grid()
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
    #plt.legend(prop={'size':14}, bbox_to_anchor=(0.1, 0), loc=3,frameon=False)
    plt.ylim(0, 3)
    plt.xlim(start, end)


    plt.savefig('compareheatmap_' + str(start) + '_' + str(end), bbox_inches='tight')
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



