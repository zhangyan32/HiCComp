import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import random
from time import gmtime, strftime
random.seed(321)
def readMatrix(filename, total_length):
    print "reading Rao's HiC "
    infile = open(filename).readlines()
    print len(infile)
    HiC = np.zeros((total_length,total_length)).astype(np.int16)
    percentage_finish = 0
    for i in range(0, len(infile)):
        if (i %  (len(infile) / 10)== 0):
            print 'finish ', percentage_finish, '%'
            percentage_finish += 10
        nums = infile[i].split('\t')
        x = int(nums[0])
        y = int(nums[1])
        val = int(float(nums[2]))

        HiC[x][y] = val
        HiC[y][x] = val
    return HiC



def main():
    chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

    cells = ['GM12878_replicate', 'GM12878_primary', 'IMR90']
    chrN_start = 19
    chrN_end = 19


    for chrN in range(chrN_start,chrN_end+1, 1):
        print "working on chr ", chrN
        matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/K562MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy'
        HiCsample = np.load(gzip.GzipFile(matrix_name + '.gz', "r"))
        base = np.mean(HiCsample) 
        for cell in cells:
      
            matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy'

            HiCsample = np.load(gzip.GzipFile(matrix_name + '.gz', "r"))
            mean = np.mean(HiCsample)
            rate = base / mean
            path = '/home/zhangyan/triplet_loss'
            new_matrix = np.zeros(HiCsample.shape)
            p = 0
            percent = HiCsample.shape[0] / 10
            for i in range(0, HiCsample.shape[0]):
                if (i % percent == 0):
                    print p, ", ", strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    p += 1
                for j in range(0, HiCsample.shape[1]):
                    val = 0
                    for k in range(0, HiCsample[i][j]):
                        if (random.random() < rate):
                            val += 1
                    new_matrix[i][j] = val
                    new_matrix[j][i] = val
            f = gzip.GzipFile('/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy.gz', "w")
            np.save(f, new_matrix)
            f.close()


            
if __name__ == "__main__":
    main()