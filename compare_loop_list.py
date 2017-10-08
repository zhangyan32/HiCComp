import numpy as np
import copy

import matplotlib.pyplot as plt
import os
def readLoop(path, chrN, min_size = 0, max_size = 500, resolution = 10000):
    infile = open(path).readlines()
    looplist = []
    for j in range(1, len(infile)):
        line = infile[j]
        words = line.split()
        aLoop = []
        
        if ('X' in words[0]):
            words[0] = '23'
        if ('X' in words[3]):
            words[3] = '23'
        if (words[0] != words[3]):
            print "interchrome..."
            continue
        if (int(words[0])!=chrN):
            continue
        aLoop.append(int(words[0]))
        start = int(words[1])/resolution
        end = int(words[4])/resolution
        size = abs(end - start)
        if (size < min_size or size > max_size):
            continue
        if (end < start):
            aLoop.append(end)
            aLoop.append(start)
        else:
            aLoop.append(start)
            aLoop.append(end)
        looplist.append(aLoop)
    print "length of looplist is ", len(looplist)
    return looplist

entire_looplist = {}
for chrN in range(18, 19):
	looplist1 = readLoop('/home/zhangyan/looplist/GSE63525_K562_HiCCUPS_looplist.txt' , chrN)
	
	looplist2 = readLoop('/home/zhangyan/looplist/GSE63525_GM12878_primary_HiCCUPS_looplist.txt' , chrN)
	looplist3 = readLoop('/home/zhangyan/looplist/GSE63525_GM12878_replicate_HiCCUPS_looplist.txt' , chrN)

		#print looplist1, looplist2
	looplist1 = sorted(looplist1, key=lambda x:(x[1], x[2]))
	looplist2 = sorted(looplist2, key=lambda x:(x[1], x[2]))
	looplist3 = sorted(looplist3, key=lambda x:(x[1], x[2]))
	HT = {}
	for line in looplist1:
		HT[tuple(line)] = [1]
	for line in looplist2:
		if (tuple(line) in HT):
			HT[tuple(line)].append(2)
		else:
			HT[tuple(line)] = [2]
	for line in looplist3:
		continue
		if (tuple(line) in HT):
			HT[tuple(line)].append(3)
		else:
			HT[tuple(line)] = [3]
	keys = HT.keys()
	keys = sorted(keys, key=lambda x:(x[1], x[2]))
	threshold = 5
	finished = False
	while(not finished):
		
		keys = copy.deepcopy(HT.keys())
		#print "another cycle", len(keys)
		keys = sorted(keys, key=lambda x:(x[1], x[2]))
		finished = True
		for i in range(0, len(keys)):
			k = 1
			startover = False
			#print i, i+k, len(keys), keys[i], keys[i+k], len(keys[i+k])
			while(i+k < len(keys) and keys[i+k][1] - keys[i][1] < threshold):

				if (i + k >= len(keys)):
					break
				if (abs(keys[i+k][1] - keys[i][1]) <= threshold and abs(keys[i+k][2] - keys[i][2]) <= threshold ):
					#print "merging ", keys[i], keys[i+k]
					finished = False
					newkey = (keys[i][0], (keys[i][1] + keys[i+k][1])/2, (keys[i][2] + keys[i+k][2])/2)
					newValue = HT[keys[i]] + HT[keys[i+k]]

					del HT[keys[i]]
					if (keys[i] != keys[i+k]):
						del HT[keys[i+k]]
					startover = True
					HT[newkey] = newValue
					break
				k += 1
			if (startover == True):
				break
	keys = HT.keys()
	print chrN, 'after merging', len(keys)
	keys = sorted(keys, key=lambda x:(x[1], x[2]))
	for key in keys:
		entire_looplist[key] = HT[key]
np.save('/home/zhangyan/looplist/GM12878_and_K562_loop.npy', entire_looplist)
#HT = np.load('/home/zhangyan/looplist/comprehensive_looplist.npy').item()

print entire_looplist

group1 = 0
group2 = 0
group3 = 0
intersection = 0
print "Union Length is ", len(entire_looplist.keys())
for k in entire_looplist:
	if (len(entire_looplist[k]) == 1):
		intersection += 1
print "intersection Length is ", intersection

