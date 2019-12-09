

import numpy as np
import sys
import argparse
import subprocess
import time
import math
import os

def convert_char_to_int_in_genome(genome):
	gen = []
	template_dict = {'A':0,'C':1, 'G':2,'T':3, 'Z':5}
	for i in genome:
		gen.append(template_dict[i])
	return gen


def readGenome(f):
	genome = ''
	for line in f:
		if not line[0] == '>':
			genome += line.rstrip()
	return genome


def RefGenomePreparation(f):
	genome = readGenome(f)
	gen = ''
	for s in genome:
		if s == 'N' or s == 'n':
			s = 'Z'		
		gen += s.upper()	
	genome = convert_char_to_int_in_genome(gen)
	genome = np.asarray(genome,np.uint8)	
	np.save('genome_uppercase.npy',genome, allow_pickle=False)
	gen = gen+'$'
	file = open('genome_uppercase.txt','w')
	file.write(gen) 
	file.close()	
	return gen


def FMIndexViaSA(genome, SA):
	L = []
	F = []
	for si in SA:
		F.append(genome[si])
		if si == 0:
			L.append('$')
		else:
			L.append(genome[si-1])
	return  F, L


def build_F(tally):
	# F array stores the cummulative sum of alphabets in the sequence A, C, G and T
	F = np.zeros(5,np.uint32)   # 4 + 1, where 4 is the number of alphabets and remaining one is the cummulative sum of all the characters.
	F[0] = 1
	for i in range(1,len(F)):
		F[i] = F[i-1] + tally[-1][i-1]	
	np.save('F_FMIndex.npy',F, allow_pickle=False)
	print(F)


def build_tally_matrix(F, L):
	template_dict = {'A':0,'C':1, 'G':2,'T':3, 'Z':5, '$':5}
	L_int = []
	for i in L:
		L_int.append(template_dict[i])
	tally = np.zeros((len(L),4),np.uint32)
	if(L_int[0] != 5):
		tally[0][L_int[0]] = 1
	for i in range(1,len(L_int)):
		tally[i] = tally[i-1]		
		if(L_int[i] != 5):
			tally[i][L_int[i]] += 1
	np.save('Tally.npy',tally, allow_pickle=False)
	# for i in range(len(F)):
	# 	# print(F[i],L[i], '	', tally[i][0], tally[i][1], tally[i][2], tally[i][3], '	',SA[i])
	# 	print(F[i],L[i],'{', tally[i][0],',', tally[i][1],',', tally[i][2],',', tally[i][3], ',', tally[i][4],'},', end='')
	build_F(tally)


def main():
	print('\n');print('---------------------------------------------------------------------------')
	print('This process is not parallelised, and hence, takes time. Please wait or use our pre uploaded data.')
	np.set_printoptions(threshold=sys.maxsize)

	p = subprocess.run(['pwd'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/'

	parser = argparse.ArgumentParser(description="Preprocessing module for REPUTE : REad maPper  for  heterogeneoUs sysTEms.")
	parser.add_argument("Ref_Gen_Name", help="Provide the reference genome file name in fasta format.", type=str)
	args = parser.parse_args()	

	start_time = time.time()
	with open(p + args.Ref_Gen_Name,'r') as f:
		genome = RefGenomePreparation(f)
	#------------------------------------------------------------------------------------------
	subprocess.call(['./suftest','genome_uppercase.txt']);
	SA = []
	with open('SA.txt') as f:
		for line in f:
			SA.append(line.rstrip())
	SA = np.asarray(SA,dtype=np.uint32)
	np.save('SA.npy',SA, allow_pickle=False)
	subprocess.call(['rm','SA.txt']);
	#------------------------------------------------------------------------------------------
	F,L = FMIndexViaSA(genome, SA)
	build_tally_matrix(F, L)
	subprocess.call(['rm','genome_uppercase.txt']);
	total_time = time.time() - start_time
	print('Length of reference genome =',len(genome)-1)
	print('Time for preprocessing =', total_time, 's')


if __name__ == '__main__':
	main()