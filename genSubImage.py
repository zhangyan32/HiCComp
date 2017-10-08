import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import random
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
    cells = ['GM12878_replicate', 'GM12878_primary', 'IMR90', 'K562']
    chrN_start = 1
    chrN_end = 8
    zero_index = [[],[]]
    for i in range(0, subImage_size):
        for j in range(i, subImage_size):
            zero_index[0].append(i)
            zero_index[1].append(j)
    for cell in cells:
        result = []
        index = []
        for chrN in range(chrN_start,chrN_end+1, 1):
            HiCfile = '/home/zhangyan/rawHiC10k/'+cell+'MAPQGE30chr'+str(chrN)+'_10kb.RAWobserved'
            #matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'_HindIII_chr'+str(chrN)+'_10kb.npy'
            #matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy'
            if (cell == 'K562'):
                matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved.npy'
            else:
                matrix_name = '/home/zhangyan/SRHiC_samples/HiCmatrix/'+cell+'MAQPGE30chr'+str(chrN)+'_10kb.RAWobserved_downsampletoK562.npy'
            if os.path.exists(matrix_name + '.gz'):
                print 'loading zip form of ', matrix_name
                HiCsample = np.load(gzip.GzipFile(matrix_name + '.gz', "r"))
            elif os.path.exists(matrix_name):
                print 'loading ', matrix_name
                HiCsample = np.load(matrix_name)
            else:
                print matrix_name, 'not exist, creating'
                print HiCfile           
                HiCsample = readMatrix(HiCfile, (chrs_length[chrN-1]/10000 + 1))
                #HiCsample = np.loadtxt('/home/zhangyan/private_data/IMR90.nodup.bam.chr'+str(chrN)+'.10000.matrix', dtype=np.int16)
                print HiCsample.shape
                np.save(matrix_name, HiCsample)
            print cell
            print HiCsample.shape
            print np.mean(HiCsample)

            path = '/home/zhangyan/triplet_loss'
            if not os.path.exists(path):
                os.makedirs(path)
            total_loci = chrs_length[chrN-1]/10000
            for i in range(0, total_loci, step):

                if (i + subImage_size >= total_loci):
                    continue
                subImage = HiCsample[i:i+subImage_size, i:i+subImage_size]
                subImage[zero_index] = 0
                result.append([subImage,])
                index.append((cell, chrN, i, i))

        result = np.array(result)
        print result.shape
        result = result.astype(np.int16)
        np.save(path + '/'+cell+'size'+str(subImage_size)+'_diag_step1_chr'+str(chrN_start)+'_' + str(chrN_end), result)
        index = np.array(index)
        np.save(path + '/'+cell+'size'+str(subImage_size)+'_diag_step1_index_chr'+str(chrN_start)+'_' + str(chrN_end), index)




    #heatmap = plt.imshow(HiCsample[i], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 25)
    #plt.show()


  
if __name__ == "__main__":
    main()