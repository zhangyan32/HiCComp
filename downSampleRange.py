import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
from time import gmtime, strftime
import random
random.seed(321)
def downsample(matrix, rate):
    #result = np.zeros(matrix.shape).astype(np.int16)
    result = np.copy(matrix)
    result[:,:] = 0
    percent = int(matrix.shape[0]/100)
    p = 0
    print matrix.shape
    for i in range(0, matrix.shape[0]):
        if (i % percent == 0):
            print p, ", ", strftime("%Y-%m-%d %H:%M:%S", gmtime())
            p += 1
        for x in range(0, matrix.shape[2]):
            for y in range(0, matrix.shape[3]):
                new_result = 0
                for n in range(0, matrix[i][0][x][y]):
                    if (random.random() < rate):
                        new_result += 1
                result[i][0][x][y] = new_result
    return result





def main():
    #cells =['GM12878_combined','GM12878','GM12878_replicate',  'HMEC',  'HUVEC',  'IMR90',  'K562',  'KBM7',  'NHEK']
    chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]
    #cells =[ 'HUVEC']
    #cells =['K562', 'IMR90']
    #cells =['NHEK', 'KBM7',]
    cells = ['K562','GM12878_replicate', 'GM12878_primary']
    cells = ['GM12878_replicate']
    cells = ['IMR90']
    #cells = ['GM12878']
    #cells = ['K562']
    cells = ['GM12878_replicate', 'K562', 'IMR90']
    step = 1
    subImage_size = 100
    cells = ['GM12878_primary']
    cells = ['GM12878_replicate', 'GM12878_primary', 'IMR90']
    #cells = ['IMR90']
    #cells = ['GM12878_replicate']
    chrN_start = 18
    chrN_end = 18
    path = '/home/zhangyan/triplet_loss'
    low_raw = np.load(path + '/K562size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy')
    base = np.mean(low_raw)
    print base
    for cell in cells:
        
        raw = np.load(path + '/'+cell+'size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '.npy')
        print cell
        rate = base/np.mean(raw)
        print np.mean(raw)
        result = downsample(raw, rate)
        np.save(path + '/'+cell+'size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end) + '_downtoK562.npy', result)









if __name__ == "__main__":
    main()