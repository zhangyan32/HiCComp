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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
path = '/home/zhangyan/triplet_loss'
chrN = 18
chrN_start = chrN     
chrN_end = chrN


subImage_size = 100
anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)
print "shape of the samples", anchor.shape

if (0):
    Pearson_pos = []
    Pearson_neg = []
    Pearson_K562 = []
    correct = 0
    incorrect = 0
    for i in range(0, 50):
        Pearson_pos.append(0)
        Pearson_neg.append(0)
        Pearson_K562.append(0)

    for i in range(0, anchor.shape[0]):
        pos = pearsonr(anchor[i].flatten(), positive[i].flatten())[0]
        neg = pearsonr(anchor[i].flatten(), negative[i].flatten())[0]
        _K562 = pearsonr(anchor[i].flatten(), K562[i].flatten())[0]
        print pos, neg, _K562
        if (pos > _K562):
            correct +=1
        else:
            incorrect +=1
        Pearson_pos.append(pos)
        Pearson_neg.append(neg)
        Pearson_K562.append(_K562)
    for i in range(0, 50):
        Pearson_pos.append(0)
        Pearson_neg.append(0)
        Pearson_K562.append(0)
    Pearson_pos = np.nan_to_num(np.array(Pearson_pos))
    Pearson_neg = np.nan_to_num(np.array(Pearson_neg))
    Pearson_K562 = np.nan_to_num(np.array(Pearson_K562))
    print correct, incorrect
    np.save(path + '/pearson_pos_vs_anchor_size100_chr'+str(chrN_start)+'.npy', Pearson_pos)
    np.save(path + '/pearson_neg_vs_anchor_size100_chr'+str(chrN_start)+'.npy', Pearson_neg)
    np.save(path + '/pearson_K562_vs_anchor_size100_chr'+str(chrN_start)+'.npy', Pearson_K562)
    print Pearson_K562.shape

    sys.exit()

batch_size = anchor.shape[0]/200 + 1
training_set = data.TensorDataset(torch.from_numpy(anchor), torch.from_numpy(np.zeros(anchor.shape[0])))
train_loader_anchor = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
print "number of batch is ", len(train_loader_anchor)
tenth_bath = len(train_loader_anchor) / 10
training_set = data.TensorDataset(torch.from_numpy(positive), torch.from_numpy(np.zeros(positive.shape[0])))
train_loader_positive = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(negative), torch.from_numpy(np.zeros(negative.shape[0])))
train_loader_negative = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(K562), torch.from_numpy(np.zeros(negative.shape[0])))
train_loader_K562 = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

def d(M1, M2):
    return np.sqrt(np.sum((M1 - M2) ** 2))


Net = model.TripletNetwork()

if use_gpu:
    Net = Net.cuda()

optimizer = optim.SGD(Net.parameters(), lr = 0.001, momentum=0.9)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
Net.eval()
#Net.load_state_dict(torch.load(path + '/triplet_losstriplet_chr1_8_epoch_330'))
#Net.load_state_dict(torch.load(path + '/tripletMarginLoss_reg_chr18_epoch_size100_350'))
Net.load_state_dict(torch.load(path + '/tripletMarginLoss_dropout_9_3_3_L2_7latent_chr1_8_epoch_size100_100'))


running_loss1 = 0.0
running_loss2 = 0.0
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
    output_anchor, output_pos, output_neg, output_K562 = Net(anchorHiC, positiveHiC, negativeHiC, K562HiC)
    loss = triplet_loss(output_anchor, output_pos, output_neg)
    running_loss1 += loss.data[0]
    loss = triplet_loss(output_anchor, output_pos, output_K562)
    running_loss2 += loss.data[0]
    print "end prediction"
    if (i == 0):
        #print "create"
        output_anchor_npy = output_anchor.cpu().data.numpy()
        output_pos_npy = output_pos.cpu().data.numpy()
        output_neg_npy = output_neg.cpu().data.numpy()
        output_K562_npy = output_K562.cpu().data.numpy()
    else:
        #print "concat", output_anchor.cpu().data.numpy().shape
        output_anchor_npy = np.concatenate((output_anchor_npy, output_anchor.cpu().data.numpy()), axis=0)
        output_pos_npy = np.concatenate((output_pos_npy, output_pos.cpu().data.numpy()), axis=0)
        output_neg_npy = np.concatenate((output_neg_npy, output_neg.cpu().data.numpy()), axis=0)
        output_K562_npy = np.concatenate((output_K562_npy, output_K562.cpu().data.numpy()), axis=0)       
    #print output_anchor_npy.shape
    if (i == len(train_loader_anchor) - 1):
        print "loss on IMR90", running_loss1/i, "loss on K562", running_loss2/i, 
prefix = 'chr'+str(chrN_start)+'_test_size100_triplet_reg_'


np.save(path + '/' + prefix + "anchor_step1.npy", output_anchor_npy)
np.save(path + '/' + prefix + "pos_step1.npy", output_pos_npy)
np.save(path + '/' + prefix + "neg_step1.npy", output_neg_npy)
np.save(path + '/' + prefix + "output_K562_step1.npy", output_K562_npy)

for i in range(1000, output_anchor_npy.shape[0],1):
    print "-----------"
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



