import torch
import torchvision.models as models
from matplotlib import pyplot as plt
import model
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from time import gmtime, strftime
def plot_kernels(tensor, num_cols=4):
    print "drawing ", tensor.shape
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols * 10,num_rows*10))

    for i in range(0, tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        plt.imshow(tensor[i][0],  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.axis('off')


    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def plot_featuremap(tensor, input_matrix, num_cols=4, caller='', layers = ''):
    print "drawing ", tensor.shape
    num_kernels = tensor.shape[1]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols * 10,num_rows * 10))

    ax1 = fig.add_subplot(num_rows,num_cols,num_kernels + 1)
    plt.imshow(input_matrix,  cmap=plt.cm.jet, interpolation='nearest', origin='lower', vmin = 0, vmax = 50)
    plt.colorbar()
    plt.axis('off')
    for i in range(0, tensor.shape[1]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        plt.imshow(tensor[0][i],  cmap=plt.cm.jet, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.axis('off')


    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(caller + '_sample_2400_occlude_' + layers, bbox_inches='tight' )
    #plt.show()
    plt.close()

path = '/home/zhangyan/triplet_loss'
Net = model.TripletNetwork()  
Net.eval()
#Net.load_state_dict(torch.load(path + '/tripletMarginLoss_chr1_8_epoch_size100_1550'))
Net.load_state_dict(torch.load(path + '/tripletMarginLoss_dropout_9_3_3_L2_7latent_chr1_8_epoch_size100_100'))
#mm = Net.double()
mm = Net
print mm
filters = mm.modules
print filters
body_model = [i for i in mm.children()][0]
layer1 = body_model
tensor = layer1.weight.data.numpy()
bias = layer1.bias.data.numpy()
print bias
plot_kernels(tensor)


path = '/home/zhangyan/triplet_loss'
chrN = 18
chrN_start = chrN     
chrN_end = chrN


subImage_size = 100
start = 2400
end = start + 1
anchor = np.load(path + '/GM12878_primarysize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
positive = np.load(path + '/GM12878_replicatesize'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
negative = np.load(path + '/IMR90size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy').astype(np.float32)[start:end]
K562 = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end)+ '.npy').astype(np.float32)[start:end]

batch_size = anchor.shape[0]
training_set = data.TensorDataset(torch.from_numpy(anchor), torch.from_numpy(np.zeros(anchor.shape[0])))
train_loader_anchor = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
print "number of batch is ", len(train_loader_anchor)
tenth_bath = max(1,  len(train_loader_anchor) / 10)
training_set = data.TensorDataset(torch.from_numpy(positive), torch.from_numpy(np.zeros(positive.shape[0])))
train_loader_positive = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(negative), torch.from_numpy(np.zeros(negative.shape[0])))
train_loader_negative = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

training_set = data.TensorDataset(torch.from_numpy(K562), torch.from_numpy(np.zeros(negative.shape[0])))
train_loader_K562 = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)

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

    print "start prediction"
    output_anchor, output_pos, output_neg, output_K562 = Net(anchorHiC, positiveHiC, negativeHiC, K562HiC)

    print "end prediction"

    output_anchor_npy = output_anchor.cpu().data.numpy()
    output_pos_npy = output_pos.cpu().data.numpy()
    output_neg_npy = output_neg.cpu().data.numpy()
    output_K562_npy = output_K562.cpu().data.numpy()
  
    print output_anchor_npy.shape
print output_anchor_npy.shape
plot_featuremap(output_anchor_npy, anchor[0][0], 4,"anchor", "1_conv")
plot_featuremap(output_K562_npy, K562[0][0], 4,"K562", "1_conv")
plot_featuremap(output_pos_npy, positive[0][0], 4,"pos", "1_conv")

