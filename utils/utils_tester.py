import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle


def familywise_db_tester():
	len_db = {}
	len_db[50] = 1787
	len_db[100] = 980
	len_db[200] = 549
	
	# match this data
	len_db_seqs = {}
	len_db_seqs[50] = 388102
	len_db_seqs[100] = 332315
	len_db_seqs[200] = 272595
	ans = True
	for min_no_of_seq in [50, 100, 200]:
		file_path = './../data/db_' + str(min_no_of_seq) +'_pickle'
		input_p = open(file_path, 'rb')
		db_ = pickle.load(input_p)
		ans = ans and (len_db[min_no_of_seq] == len(db_.keys()))
		no_of_seq = 0
		for i in db_.keys():
			no_of_seq += len(db_[i])
		ans = ans and (len_db_seqs[min_no_of_seq] == no_of_seq)
		input_p.close()
	if ans:
		print("Passed the db size tests. ")
	else:
		print("Failed the db size tests. ")

def map_creator_tester():
	file_path = './../data/amino_acid_map_pickle'
	input_p = open(file_path, 'rb')
	amino_acid_map = pickle.load(input_p)
	input_p.close()

	ans = True
	ans = ans and (amino_acid_map['Y'] == 20)
	ans = ans and (amino_acid_map['M'] == 11)
	ans = ans and (amino_acid_map['S'] == 16)

	file_path = './../data/families_map_pickle'
	input_p = open(file_path, 'rb')
	families_map = pickle.load(input_p)
	input_p.close()

	ans = True
	ans = ans and (families_map['Class-II aminoacyl-tRNA synthetase family'] == 1)
	ans = ans and (families_map['RRF family'] == 87)
	ans = ans and (families_map['TGF-beta family'] == 510 )

	if ans:
		print("Passed the mapping tests. ")
	else:
		print("Failed the mapping tests. ")


familywise_db_tester()
map_creator_tester()
