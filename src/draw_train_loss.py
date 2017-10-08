
import matplotlib

import sys
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import poisson
import os.path
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as st
import os
import urllib
import gzip
import cPickle
import sys
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import gzip
import math


import csv
def readcsvdata(file_name):
	training = []
	validation = []
	x = []
	data = csv.reader(open(file_name))

	for line in data:
		x.append(line[0])
		training.append(line[1])
		validation.append(line[3])
	return x, training, validation

if (1):
	plt.figure(figsize=(8,6))
	ax1 = plt.plot()

	x, training, validation = readcsvdata("9_7latent.txt")
	plt.plot(x, validation, color='r', linestyle = '-', label='Number of Convolution Layers = 1',linewidth=3)
	#plt.plot(x, validation1, color='r', linestyle = '--',linewidth=3)

	x, training, validation = readcsvdata("9_3_7latent.txt")
	plt.plot(x, validation, color='c', linestyle = '-', label='Number of Convolution Layers = 2',linewidth=3)
	#plt.plot(x, validation, color='c', linestyle = '--',linewidth=3)

	x, training, validation = readcsvdata("9_3_3_7latent.txt")
	plt.plot(x, validation, color='b', linestyle = '-', label='Number of Convolution Layers = 3',linewidth=3)
	#plt.plot(x, validation, color='b', linestyle = '--',linewidth=3)
	x, training, validation = readcsvdata("9_3_3_3_7latent.txt")
	plt.plot(x, validation, color='k', linestyle = '-', label='Number of Convolution Layers = 4',linewidth=3)
	#plt.plot(x, validation, color='k', linestyle = '--',linewidth=3)







	plt.xticks(fontsize = 24)
	plt.yticks(fontsize = 24)
	#plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
	plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.6), loc=3,frameon=False)
	#plt.ylim(0, 0.25)
	plt.ylabel('Triplet Loss', fontsize=24)
	plt.xlabel('Epoches', fontsize=24)
	#plt.ylim(0, 350)
	plt.gca().yaxis.grid(True)
	#plt.show()
	plt.savefig("Conv_Layers.png" ,bbox_inches='tight')
	plt.close()

	#sys.exit()

if (0):
	plt.figure(figsize=(8,6))
	ax1 = plt.plot()

	x, training, validation = readcsvdata("9_3_3_3latent.txt")
	plt.plot(x, validation, color='r', linestyle = '-', label='Number of Latent Variables = 3',linewidth=3)
	#plt.plot(x, validation1, color='r', linestyle = '--',linewidth=3)

	x, training, validation = readcsvdata("9_3_3_5latent.txt")
	plt.plot(x, validation, color='c', linestyle = '-', label='Number of Latent Variables = 5',linewidth=3)
	#plt.plot(x, validation, color='c', linestyle = '--',linewidth=3)

	x, training, validation = readcsvdata("9_3_3_7latent.txt")
	plt.plot(x, validation, color='b', linestyle = '-', label='Number of Latent Variables = 7',linewidth=3)
	#plt.plot(x, validation, color='b', linestyle = '--',linewidth=3)
	x, training, validation = readcsvdata("9_3_3_9latent.txt")
	plt.plot(x, validation, color='k', linestyle = '-', label='Number of Latent Variables = 9',linewidth=3)
	#plt.plot(x, validation, color='k', linestyle = '--',linewidth=3)



	x, training, validation = readcsvdata("9_3_3_11latent.txt")
	plt.plot(x, validation, color='m', linestyle = '-', label='Number of Latent Variables = 11',linewidth=3)
	#plt.plot(x, validation, color='m', linestyle = '--',linewidth=3)




	plt.xticks(fontsize = 24)
	plt.yticks(fontsize = 24)
	#plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
	plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.6), loc=3,frameon=False)
	#plt.ylim(0, 0.25)
	plt.ylabel('Triplet Loss', fontsize=24)
	plt.xlabel('Epoches', fontsize=24)
	#plt.ylim(0, 350)
	plt.gca().yaxis.grid(True)
	#plt.show()
	plt.savefig("N_Latent_Var.png" ,bbox_inches='tight')
	plt.close()

	#sys.exit()

if (0):
	plt.figure(figsize=(8,6))
	ax1 = plt.plot()


	x, training, validation = readcsvdata("9_3_3_5latent.txt")
	plt.plot(x, training, color='b', linestyle = '-', label='training loss',linewidth=3)
	plt.plot(x, validation, color='b', linestyle = '--', label='validation loss',linewidth=3)


	plt.xticks(fontsize = 24)
	plt.yticks(fontsize = 24)
	#plt.legend(prop={'size':14}, bbox_to_anchor=(0.3, 0.7), loc=3,frameon=False)
	plt.legend(prop={'size':14}, bbox_to_anchor=(0.4, 0.6), loc=3,frameon=False)
	#plt.ylim(0, 0.25)
	plt.ylabel('Triplet Loss', fontsize=24)
	plt.xlabel('Epoches', fontsize=24)
	#plt.ylim(0, 350)
	plt.gca().yaxis.grid(True)
	#plt.show()
	plt.savefig("training_vs_testing_loss.png" ,bbox_inches='tight')
	plt.close()
