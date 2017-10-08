import numpy as np
import model

from time import gmtime, strftime
use_gpu = 1
import matplotlib.pyplot as plt

path = '/home/zhangyan/triplet_loss'
chrN_start = 21     
chrN_end = 21

anchor = np.load(path + '/GM12878_primary_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
positive = np.load(path + '/GM12878_replicate_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)
negative = np.load(path + '/IMR90_diag_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy').astype(np.float32)

print np.max(anchor)