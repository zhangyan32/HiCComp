import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]


def readState(filename, N):
	f = open(filename).readlines()
	print len(f)

	state = np.zeros(chrs_length[N-1])
	print state.shape
	#state[:] = 8


	HT = {'WE': 4, 'E': 3, 'R': 0, 'PF': 2, 'T': 6, 'TSS': 1, 'CTCF': 5}
	HT = {'WE': 0, 'E': 0, 'R': 0, 'PF': 0, 'T': 0, 'TSS': 0, 'CTCF': 1}
	for line in f:
		words = line.split('\t')
		try:
			chrN = int(words[0].split('r')[1])
		except:
			continue
		if (chrN != N):
			continue
		start = int(words[1])
		end = int(words[2])

		marker = HT[words[3]]
		try:
			state[start:end] = marker
		except:
			print start, end
			continue
	return np.array(state)

N = 17
path = '/home/zhangyan/triplet_loss'
K562_state = readState(path + '/wgEncodeAwgSegmentationCombinedK562.bed', N)
np.save(path + '/state_bp_K562_chr' + str(N), K562_state)
GM12878_state = readState(path + '/wgEncodeAwgSegmentationCombinedGm12878.bed', N)
np.save(path + '/state_bp_Gm12878_chr' + str(N), GM12878_state)

diff = K562_state - GM12878_state
share = K562_state + GM12878_state
K562_non_zero  = np.zeros(chrs_length[N-1]/10000+1)
GM12878_non_zero  = np.zeros(chrs_length[N-1]/10000+1)
bin_diff = np.zeros(chrs_length[N-1]/10000+1)
bin_share = np.zeros(chrs_length[N-1]/10000+1)
for i in range(0, bin_diff.shape[0]):
	bin_diff[i] = np.count_nonzero(diff[i*10000:(i+1)*10000])
	bin_share[i] = np.count_nonzero(share[i*10000:(i+1)*10000])
	K562_non_zero[i] = np.count_nonzero(K562_state[i*10000:(i+1)*10000])
	GM12878_non_zero[i] = np.count_nonzero(GM12878_state[i*10000:(i+1)*10000])

for i in range(0, bin_diff.shape[0]):
	if (bin_share[i] != 0):
		print i, bin_diff[i], bin_share[i], bin_diff[i]/1.0/bin_share[i]
print bin_diff
np.save('state_diff_K562_GM12878_CTCF_chr' + str(N), bin_diff)
np.save('state_share_K562_GM12878_CTCF_chr' + str(N), bin_share)