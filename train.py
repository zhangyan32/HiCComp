

import numpy as np
import model
import torch.nn as nn
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
use_gpu = 1
batch_size = 256
subImage_size = 100
path = '/home/zhangyan/triplet_loss'
chrN_start = 1     
chrN_end = 8

anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)

print K562.shape
sys.exit()
print anchor.shape, positive.shape, K562.shape
print np.mean(anchor), np.mean(positive), np.mean(negative), np.mean(K562)


if (0): 
    chrN_start = 18     
    chrN_end = 18
    anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
    positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)
    negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)


subImage_size = 100
training_set = data.TensorDataset(torch.from_numpy(anchor), torch.from_numpy(np.zeros(anchor.shape[0])))
train_loader_anchor = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
print "number of batch is ", len(train_loader_anchor)
tenth_bath = len(train_loader_anchor) / 10
training_set = data.TensorDataset(torch.from_numpy(positive), torch.from_numpy(np.zeros(positive.shape[0])))
train_loader_positive = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(negative), torch.from_numpy(np.zeros(negative.shape[0])))
train_loader_negative = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(K562), torch.from_numpy(np.zeros(K562.shape[0])))
train_loader_K562 = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

Net = model.TripletNetwork()

if use_gpu:
    Net = Net.cuda()

optimizer = optim.SGD(Net.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.01)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
Net.train()
#Net.load_state_dict(torch.load(path + '/tripletMarginLoss_chr1_8_epoch_size100_1550'))
#
#Net.load_state_dict(torch.load(path + '/tripletMarginLoss_dropout_chr1_8_epoch_size100_110'))
#Net.load_state_dict(torch.load(path + '/tripletMarginLoss_dropout_9_3_3_twoCells_chr1_8_epoch_size100_50'))
running_loss = 0.0
running_loss_validate = 0.0
reg_loss = 0.0
log = open('9_3_3_16filters_7latent.txt', 'w')
for epoch in range(0, 101):
    for i, (v1, v2, v3, v4) in enumerate(zip(train_loader_anchor, train_loader_positive, train_loader_negative, train_loader_K562)):    
        if (i == len(train_loader_anchor) - 1):
            continue 
        #print i
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
       
        optimizer.zero_grad()
        output_anchor, output_pos, output_neg, output_K562 = Net(anchorHiC, positiveHiC, negativeHiC, K562HiC)
        loss = triplet_loss(output_anchor, output_pos, output_neg) 
        #loss = (torch.pow(((output_anchor - output_pos) *(output_anchor - output_pos)).sum(), 0.5)  -   torch.pow(((output_anchor - output_neg)  * (output_anchor - output_neg)).sum(), 0.5)+1)/float(batch_size)
        #loss = 



        reg = ((output_anchor * output_anchor).sum() +(output_neg * output_neg).sum() + (output_pos * output_pos).sum()).sqrt()/float(batch_size)
        validate_loss = triplet_loss(output_anchor, output_pos, output_K562) 
        total_loss = loss
        #total_loss = loss / reg
        total_loss.backward()  




        
        optimizer.step()
        running_loss += loss.data[0]
        running_loss_validate += validate_loss.data[0]
        reg_loss += reg.data[0]
        if (i != 0 and i % tenth_bath == 0):
            pass
            #print epoch, i, running_loss/i, reg_loss/i, strftime("%Y-%m-%d %H:%M:%S", gmtime())
        
    print '-------', epoch, running_loss/i, reg_loss/i, running_loss/i+reg_loss/i,  running_loss_validate/i, strftime("%Y-%m-%d %H:%M:%S", gmtime())
    log.write(str(epoch) + ', ' + str(running_loss/i,) + ', ' +str(reg_loss/i) + ', ' +str(running_loss_validate/i) + '\n')
    running_loss = 0.0
    running_loss_validate = 0.0
    reg_loss = 0.0

    if (epoch % 10 == 0):

        torch.save(Net.state_dict(), path + '/tripletMarginLoss_dropout_9_3_3_16filters_L2_7latent_chr1_8_epoch_size100_' + str(epoch))
        pass


