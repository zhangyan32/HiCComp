import numpy as np
import model
import torch.nn as nn
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
use_gpu = 0
import matplotlib.pyplot as plt
import sys

path = '/home/zhangyan/triplet_loss'
chrN_start = 18     
chrN_end = 18

sample_number = 3000
anchor_raw = np.load(path + '/GM12878_primary_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
positive_raw = np.load(path + '/GM12878_replicate_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
negative_raw = np.load(path + '/IMR90_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)

for sample_number in range (0, 3500, 50):
    print sample_number
    anchor = anchor_raw[sample_number]
    positive = positive_raw[sample_number]
    negative = negative_raw[sample_number]

    batch_size = anchor.shape[0]

    anchor = [anchor]
    positive = [positive]
    negative = [negative]
    windows_size = 3
    importance_map_decrease = np.zeros((50, 50))
    for i in range(0, 51 - windows_size):

        z = np.copy(np.copy(anchor[0][0]))
 
        z[i:i+windows_size, :] = 0
        anchor.append([z,])
        z = np.copy(np.copy(positive[0][0]))
        z[i:i+windows_size, :] = 0
        positive.append([z,])
        z = np.copy(np.copy(negative[0][0]))
        z[i:i+windows_size, :] = 0
        negative.append([z,])

        z[:,i:i+windows_size] = 0
        anchor.append([z,])
        z = np.copy(np.copy(positive[0][0]))
        z[:,i:i+windows_size] = 0
        positive.append([z,])
        z = np.copy(np.copy(negative[0][0]))
        z[:,i:i+windows_size] = 0
        negative.append([z,])

    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)


    print anchor.shape, positive.shape, negative.shape
    #sys.exit()
    batch_size = anchor.shape[0]
    training_set = data.TensorDataset(torch.from_numpy(anchor), torch.from_numpy(np.zeros(anchor.shape[0])))
    train_loader_anchor = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
    print "number of batch is ", len(train_loader_anchor)
    tenth_bath = len(train_loader_anchor) / 10
    training_set = data.TensorDataset(torch.from_numpy(positive), torch.from_numpy(np.zeros(positive.shape[0])))
    train_loader_positive = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

    training_set = data.TensorDataset(torch.from_numpy(negative), torch.from_numpy(np.zeros(negative.shape[0])))
    train_loader_negative = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

    train_loader_K562 = train_loader_negative

    def d(M1, M2):
        return np.sqrt(np.sum((M1 - M2) ** 2))


    Net = model.TripletNetwork()

    if use_gpu:
        Net = Net.cuda()

    optimizer = optim.SGD(Net.parameters(), lr = 0.001, momentum=0.9)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    Net.eval()
    Net.load_state_dict(torch.load(path + '/triplet_maxMargin_chr1_8_epoch_400'))

    running_loss = 0.0

    for i, (v1, v2, v3, v4) in enumerate(zip(train_loader_anchor, train_loader_positive, train_loader_negative, train_loader_K562)):

        print i
        anchorHiC, lab = v1
        positiveHiC, lab = v2
        negativeHiC, lab = v3
        K562HiC, lab = v4
        anchorHiC = Variable(anchorHiC)
        positiveHiC = Variable(positiveHiC)
        negativeHiC = Variable(negativeHiC)
        K562HiC = Variable(K562HiC)
        if use_gpu:
            anchorHiC = anchorHiC.cuda()
            positiveHiC = positiveHiC.cuda()
            negativeHiC = negativeHiC.cuda()
            K562HiC = K562HiC.cuda()
        print "start prediction"
        output_anchor, output_pos, output_neg = Net(anchorHiC, positiveHiC, negativeHiC)
        #output_anchor, output_pos, output_K562 = Net(anchorHiC, positiveHiC, K562HiC)
        print "end prediction"
        output_anchor = output_anchor.cpu().data.numpy()
        output_pos = output_pos.cpu().data.numpy()
        output_neg = output_neg.cpu().data.numpy()


    original_loss =  d(output_anchor[0], output_neg[0]) - d(output_anchor[0], output_pos[0])

    n = 1
    add_times1 = np.zeros((50, 50))
    add_times2 = np.zeros((50, 50))

    importance_map1 = np.zeros((50, 50))
    importance_map2 = np.zeros((50, 50))
    for i in range(0, 51 - windows_size):
        z = np.copy(np.copy(anchor[0][0]))
        #print z.shape
        add_times1[i:i+windows_size, :] += 1
        add_times2[:, i:i+windows_size] += 1
        importance_map1[i:i+windows_size, :] += min(0, d(output_anchor[n], output_neg[n]) -  d(output_anchor[n], output_pos[n]) - original_loss)
        n += 1
        importance_map2[:, i:i+windows_size] += min(0, d(output_anchor[n], output_neg[n]) -  d(output_anchor[n], output_pos[n]) - original_loss)
        n += 1

    importance_map1 = -importance_map1 / add_times1
    importance_map2 = -importance_map2 / add_times2
    importance_map = importance_map1 * importance_map2
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
    importance_map[zeroindex] = 0
    ax1 = plt.subplot(2, 2, 4)
    ax1.title.set_text("importance")
    plt.imshow(importance_map,  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()
    plt.close() 


