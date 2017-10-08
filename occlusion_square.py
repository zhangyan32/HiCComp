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


#anchor_raw = np.load(path + '/GM12878_primary_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#positive_raw = np.load(path + '/GM12878_replicate_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#negative_raw = np.load(path + '/IMR90_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
#negative_raw = np.load(path + '/K562_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
Net = model.TripletNetwork()



optimizer = optim.SGD(Net.parameters(), lr = 0.001, momentum=0.9)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
Net.eval()

Net.load_state_dict(torch.load(path + '/tripletMarginLoss_dropout_9_3_3_L2_7latent_chr1_8_epoch_size100_100'))

def d(M1, M2):
    #return np.sqrt(np.sum((M1 - M2) ** 2))
    return np.sqrt(np.sum((M1 - M2) ** 2))

if use_gpu:
    Net = Net.cuda()
for latent_index in range(0, 1):
    for start in range(0, 7700, 50000):
        end = start + 7700
        subImage_size = 100
        anchor_raw = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
        positive_raw = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
        #negative_raw = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
        negative_raw = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)[start:end]


        feature_important_map = []
        for sample_number in range (0, anchor_raw.shape[0], 10):
            print sample_number
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


            batch_size = anchor.shape[0]/10

            print "batch size is ", batch_size, "total number of samples in each occlusion is ", anchor.shape
            #print anchor.shape, positive.shape, negative.shape
            #sys.exit()

            training_set = data.TensorDataset(torch.from_numpy(anchor), torch.from_numpy(np.zeros(anchor.shape[0])))
            train_loader_anchor = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
            #print "number of batch is ", len(train_loader_anchor)
            tenth_bath = len(train_loader_anchor) / 10
            training_set = data.TensorDataset(torch.from_numpy(positive), torch.from_numpy(np.zeros(positive.shape[0])))
            train_loader_positive = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

            training_set = data.TensorDataset(torch.from_numpy(negative), torch.from_numpy(np.zeros(negative.shape[0])))
            train_loader_negative = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

            train_loader_K562 = train_loader_negative




           
            running_loss = 0.0

            for i, (v1, v2, v3) in enumerate(zip(train_loader_anchor, train_loader_positive, train_loader_negative)):

                print i, strftime("%Y-%m-%d %H:%M:%S", gmtime())
                anchorHiC, lab = v1
                positiveHiC, lab = v2
                negativeHiC, lab = v3

                anchorHiC = Variable(anchorHiC)
                positiveHiC = Variable(positiveHiC)
                negativeHiC = Variable(negativeHiC)

                if use_gpu:
                    anchorHiC = anchorHiC.cuda()
                    positiveHiC = positiveHiC.cuda()
                    negativeHiC = negativeHiC.cuda()

                #print "start prediction"
                output_anchor, output_pos, output_neg, _ = Net(anchorHiC, positiveHiC, negativeHiC, negativeHiC)
                #output_anchor, output_pos, output_K562 = Net(anchorHiC, positiveHiC, K562HiC)
                #print "end prediction"
                if (i == 0):

                    output_anchor_npy = output_anchor.cpu().data.numpy().astype(np.float32)
                    output_pos_npy = output_pos.cpu().data.numpy().astype(np.float32)
                    output_neg_npy = output_neg.cpu().data.numpy().astype(np.float32)
                else:
                    output_anchor_npy = np.concatenate((output_anchor_npy, output_anchor.cpu().data.numpy().astype(np.float32)), axis=0)
                    output_pos_npy = np.concatenate((output_pos_npy, output_pos.cpu().data.numpy().astype(np.float32)), axis=0)
                    output_neg_npy = np.concatenate((output_neg_npy, output_neg.cpu().data.numpy().astype(np.float32)), axis=0)
         


            #original_loss =  np.maximum(0, d(output_anchor_npy[0][0], output_pos_npy[0][0]) - d(output_anchor_npy[0][0], output_neg_npy[0][0]) + 1)
            print output_anchor_npy.shape
            #original_loss =  max(0, d(output_anchor_npy[0], output_pos_npy[0]) - d(output_anchor_npy[0], output_neg_npy[0]) + 1)
            original_loss =  d(output_anchor_npy[0], output_pos_npy[0]) - d(output_anchor_npy[0], output_neg_npy[0])
            n = 1
            add_times1 = np.zeros((subImage_size, subImage_size))


            importance_map1 = np.zeros((subImage_size, subImage_size))

            for i in range(0, subImage_size+1 - windows_size):
                for j in range(0, subImage_size+1 - windows_size):
                    z = np.copy(np.copy(anchor[0][0]))
                    #print z.shape
                    add_times1[i:i+windows_size, j:j+windows_size] += 1
                    #new_loss = np.maximum(0, d(output_anchor_npy[n][0], output_pos_npy[n][0]) - d(output_anchor_npy[n][0], output_neg_npy[n][0]) + 1)
                    new_loss = d(output_anchor_npy[n], output_pos_npy[n]) - d(output_anchor_npy[n], output_neg_npy[n])
                    #new_loss =  max(0, d(output_anchor_npy[n], output_pos_npy[n]) - d(output_anchor_npy[n], output_neg_npy[n]) + 1)
                    importance_map1[i:i+windows_size, j:j+windows_size] += min(0, (original_loss - new_loss))
                    n += 1
           

            importance_map1 = importance_map1 / add_times1
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
            feature_important_map.append(importance_map1)
        np.save(path + '/revise_K562vsGM12878_chr18_size100_step10_933_7l_important_samples_addmargin_'+str(start)+'_'+str(windows_size) + '_all_latent', np.array(feature_important_map))

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


