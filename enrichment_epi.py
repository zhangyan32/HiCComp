import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]


def readState(filename):
	f = open(filename).readlines()
	print len(f)
	state_list = []
	for L in chrs_length:
		state_list.append(np.zeros((L/10000+1, 7)))
	HT = {'WE': 3, 'E': 2, 'R': 6, 'PF': 1, 'T': 5, 'TSS': 0, 'CTCF': 4}
	for line in f:
		words = line.split('\t')
		try:
			chrN = int(words[0].split('r')[1])
		except:
			continue
		start = int(words[1])
		end = int(words[2])
		loc = (start + end)/20000
		marker = HT[words[3]]
		state_list[chrN-1][loc][marker] += (end-start)
	return np.array(state_list)

np.save('state_10kb_K562', readState('/Users/Yan/Downloads/wgEncodeAwgSegmentationCombinedK562.bed'))
np.save('state_10kb_Gm12878', readState('/Users/Yan/Downloads/wgEncodeAwgSegmentationCombinedGm12878.bed'))



