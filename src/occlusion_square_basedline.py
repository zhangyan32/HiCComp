import numpy as np
import model
import torch.nn as nn
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
use_gpu = 1
import matplotlib.pyplot as plt
import sys

path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18

from scipy.stats import pearsonr
def d_pearsonr(a, p, n):
    pos = pearsonr(a.flatten(), p.flatten())[0]
    neg = pearsonr(a.flatten(), n.flatten())[0]
    return pos - neg
def d_pixel(a, p, n):
    pos = np.sum(np.abs(a-p))
    neg = np.sum(np.abs(a-n))
    return pos - neg
#anchor_raw = np.load(path + '/GM12878_primary_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#positive_raw = np.load(path + '/GM12878_replicate_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#negative_raw = np.load(path + '/IMR90_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#negative_raw = np.load(path + '/K562_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
for start in range(6000, 7700, 20000):
    end = start + 2000
    subImage_size = 100
    anchor_raw = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
    positive_raw = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
    #negative_raw = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
    negative_raw = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)[start:end]
    def d(M1, M2):
        return np.sqrt(np.sum((M1 - M2) ** 2))

    feature_important_map1 = []
    feature_important_map2 = []
    for sample_number in range (0, anchor_raw.shape[0], 10):
        print start + sample_number, strftime("%Y-%m-%d %H:%M:%S", gmtime())
        anchor = anchor_raw[sample_number]
        positive = positive_raw[sample_number]
        negative = negative_raw[sample_number]


        anchor = [anchor]
        positive = [positive]
        negative = [negative]
        windows_size = 8
        importance_map_decrease = np.zeros((subImage_size, subImage_size))
        for i in range(0, subImage_size+1 - windows_size):
            for j in range(0, subImage_size+1 - windows_size):

                z = np.copy(np.copy(anchor[0][0])).astype(np.float32)
         
                z[i:i+windows_size, j:j+windows_size] = 0
                anchor.append([z,])
                z = np.copy(np.copy(positive[0][0]))
                z[i:i+windows_size, j:j+windows_size] = 0
                positive.append([z,])
                z = np.copy(np.copy(negative[0][0]))
                z[i:i+windows_size, j:j+windows_size] = 0
                negative.append([z,])


        anchor = np.array(anchor).astype(np.float32)
        positive = np.array(positive).astype(np.float32)
        negative = np.array(negative).astype(np.float32)

        #original_loss =  np.maximum(0, d(output_anchor_npy[0][0], output_pos_npy[0][0]) - d(output_anchor_npy[0][0], output_neg_npy[0][0]) + 1)
        original_loss =  d_pearsonr(anchor[0][0], positive[0][0], negative[0][0])
        n = 1
        add_times1 = np.zeros((subImage_size, subImage_size))


        importance_map1 = np.zeros((subImage_size, subImage_size))
        importance_map2 = np.zeros((subImage_size, subImage_size))
        for i in range(0, subImage_size+1 - windows_size):
            for j in range(0, subImage_size+1 - windows_size):
                z = np.copy(np.copy(anchor[0][0]))
                #print z.shape
                add_times1[i:i+windows_size, j:j+windows_size] += 1
                #new_loss = np.maximum(0, d(output_anchor_npy[n][0], output_pos_npy[n][0]) - d(output_anchor_npy[n][0], output_neg_npy[n][0]) + 1)
                new_loss = d_pearsonr(anchor[n][0], positive[n][0], negative[n][0])
                importance_map1[i:i+windows_size, j:j+windows_size] += min(0, (original_loss - new_loss))
                importance_map2[i:i+windows_size, j:j+windows_size] += max(0, (original_loss - new_loss))
                n += 1
       

        importance_map1 = importance_map1 / add_times1
        importance_map2 = importance_map2 / add_times1
        '''
        ax1 = plt.subplot(2, 2, 1)
        ax1.title.set_text("original")
        plt.imshow(anchor[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)



        ax1 = plt.subplot(2, 2, 2)
        #ax1.title.set_text(str(d(output_anchor[i], output_pos[i]))  )
        plt.imshow(positive[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  

        ax1 = plt.subplot(2, 2, 3)
        ax1.title.set_text(str(original_loss)  )
        plt.imshow(negative[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  
        #importance_map1[zeroindex] = 0
        ax1 = plt.subplot(2, 2, 4)
        ax1.title.set_text("importance")
        plt.imshow(importance_map1,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.show()
        plt.close() 

        '''
        feature_important_map1.append(importance_map1)
        feature_important_map2.append(importance_map2)
    np.save(path + '/K562vsGM12878_chr18_size100_step10_Pearson1_important_samples_'+str(start)+'_'+str(start), np.array(feature_important_map1))
    np.save(path + '/K562vsGM12878_chr18_size100_step10_Pearson2_important_samples_'+str(start)+'_'+str(start), np.array(feature_important_map2))

'''
    zeroindex = [[],[]]
    for i in range(0, 50):
        for j in range(i, 50):
            zeroindex[0].append(i)
            zeroindex[1].append(j)
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    print '-------------------'
    print "anchor vs positive", pearsonr(anchor[0].flatten(), positive[0].flatten())[0], spearmanr(anchor[0].flatten()[0], positive[0].flatten())
    print "anchor vs negative", pearsonr(anchor[0].flatten(), negative[0].flatten())[0], spearmanr(anchor[0].flatten()[0], negative[0].flatten())
    print '####################'
    ax1 = plt.subplot(2, 2, 1)
    ax1.title.set_text("original")
    plt.imshow(anchor[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)



    ax1 = plt.subplot(2, 2, 2)
    ax1.title.set_text(str(d(output_anchor[i], output_pos[i]))  )
    plt.imshow(positive[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  

    ax1 = plt.subplot(2, 2, 3)
    ax1.title.set_text(str(d(output_anchor[i], output_neg[i]))  )
    plt.imshow(negative[0][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 100)  
    importance_map1[zeroindex] = 0
    ax1 = plt.subplot(2, 2, 4)
    ax1.title.set_text("importance")
    plt.imshow(importance_map1,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()
    plt.close() 
    '''


