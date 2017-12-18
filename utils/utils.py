import os
import pandas as pd
import numpy as np
from collections import Counter
# import matplotlib.pyplot as plt
import pickle
from nltk import ngrams

data = pd.read_table('./../data/uniprot-all.tab', sep = '\t')
data = data.dropna(axis = 0, how = 'any')
zero_100_list = [[0] * 100]

# fig, ax = plt.subplots()
# data['Protein families'].value_counts().plot(ax=ax, kind='bar')
# # plt.show()
# fig.savefig('./../data/family_freq')

data_np = data.as_matrix()
print("Data loaded and NaN values dropped, shape : ", data_np.shape)

def save_obj(obj,filename,overwrite=1):
	if(not overwrite and os.path.exists(filename)):
		return
	with open(filename,'wb') as f:
		pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
		print("File saved to " + filename)

def load_obj(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		print("File loaded from " + filename)
		return obj

def read_glove_vec_files():
	file_path = './../data/vectors_pfam.txt'
	file = open(file_path, 'r')
	word_to_glove = {}
	for line in file:
		line = line.split()	
		word = line[0]
		glove_vec = []
		for i in range(1, 101):
			glove_vec.append(float(line[i]))
		word_to_glove[word] = glove_vec
	# print(word_to_glove['end'])
	# print(word_to_glove['QMG'])
	# print(word_to_glove['MGL'])
	file.close()
	return word_to_glove

def seq_to_glove_vector(seq, word_to_glove):
	spaced_seq = " "
	for x in seq:
		spaced_seq += x + " "
	trigrams = ngrams(spaced_seq.split(), 3)
	
	tri_list0 = []
	tri_list1 = []
	tri_list2 = []

	count_gram = 0
	for gram in trigrams:
		gram_str = gram[0] + gram[1] + gram[2]
		if count_gram % 3 == 0:
			tri_list0.append(gram_str)
		if count_gram % 3 == 1:
			tri_list1.append(gram_str)
		if count_gram % 3 == 2:
			tri_list2.append(gram_str)
		count_gram += 1

	tri_list0.append('end')
	tri_list1.append('end')
	tri_list2.append('end')
	tri_list = tri_list0 + tri_list1 + tri_list2
	# print(tri_list)
	glove_vec = []
	for word in tri_list:
		glove_vec.append(word_to_glove[word])

	return glove_vec

def save_familywise_db(min_no_of_seq = 200):
	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)
	
	no_of_families = 0
	families_included = []
	counts = []
	for k in families_count.keys():
		if(families_count[k] >= min_no_of_seq):
			no_of_families += 1
			families_included.append(k)
	# store the entire data family-wise
	# this would help to divide data 
	# into three parts with stratification

	db_ = {}
	for fam in families_included:
		db_[fam] = []

	for i in range(data_np.shape[0]):
		if(data_np[i, 3] in families_included):
			temp = [data_np[i, 0], data_np[i, 2], data_np[i, 3]]
			db_[data_np[i, 3]].append(temp)

	file_path = './../data/db_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(db_, output)
		output.close()

	no_length_seq_gt_200 = {}
	counter = 0
	total_counter = 0

	no_of_families_2 = 0
	for fam in db_.keys():
		fam_seq = db_[fam]
		count_fam = 0
		included = True
		for seq_no in range(len(fam_seq)):
			data = fam_seq[seq_no]
			seq = data[1]
			len_seq = len(seq)
			if(len(seq) > 1000):
				count_fam += 1
				included = False
		counter += count_fam
		if included:
			no_of_families_2 += 1
			print("Appending")
			counts.append(len(fam_seq))
		total_counter += len(fam_seq)				
		print(count_fam, len(fam_seq), count_fam*100/len(fam_seq),fam)

	print(min_no_of_seq, " : ", counter, total_counter, counter*100/1 + total_counter)
	print(no_of_families)

	counts.sort()
	cs = 0
	for c in counts:
		cs += c
		print(c, cs)
	print(counts)
	print(no_of_families_2)
	debug = input()

def map_creator():
	amino_acid_map = {}
	amino_acid_map['A'] = 1
	amino_acid_map['C'] = 2
	amino_acid_map['D'] = 3 # aspartic acid
	amino_acid_map['E'] = 4
	amino_acid_map['F'] = 5
	amino_acid_map['G'] = 6
	amino_acid_map['H'] = 7
	amino_acid_map['I'] = 8
	amino_acid_map['K'] = 9
	amino_acid_map['L'] = 10
	amino_acid_map['M'] = 11
	amino_acid_map['N'] = 12
	amino_acid_map['P'] = 13
	amino_acid_map['Q'] = 14
	amino_acid_map['R'] = 15
	amino_acid_map['S'] = 16
	amino_acid_map['T'] = 17
	amino_acid_map['U'] = 18 # Q9Z0J5 - confused with v ?
	amino_acid_map['V'] = 18
	amino_acid_map['W'] = 19
	amino_acid_map['Y'] = 20
	amino_acid_map['X'] = 21 # Q9MVL6 - undetermined
	amino_acid_map['B'] = 22 # asparagine/aspartic acid
	amino_acid_map['Z'] = 23 # glutamine/glutamic acid P01340

	families = []
	for i in range(data_np.shape[0]):
		families.append(data_np[i, 3])
	families_count = Counter(families)

	families_map = {}
	counter = 0
	
	for k, v in families_count.most_common():
		counter += 1
		families_map[k] = counter
	
	"""
	Class-II aminoacyl-tRNA synthetase family 3729
	3729 1
	RRF family 764
	764 87
	TGF-beta family 213
	213 510
	"""

	file_path = './../data/amino_acid_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(amino_acid_map, output)
		output.close()

	file_path = './../data/families_map' +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(families_map, output)
		output.close()

def seq_merger_one_file():
	file_path = "./../data/all_seq_in_one.txt"
	if(not os.path.isfile(file_path)):
		tri_seq_tot_0 = ""
		tri_seq_tot_1 = ""
		tri_seq_tot_2 = ""

		for i in range(data_np.shape[0]):
		# for i in range(2):
			if(i%1000 == 0):
				print("Itertion number : ", i)
			seq = str(data_np[i, 2])
			spaced_seq = " "
			for x in seq:
				spaced_seq += x + " "
			trigrams = ngrams(spaced_seq.split(), 3)
			tri_seq_list_0 = []
			tri_seq_list_1 = []
			tri_seq_list_2 = []

			counter_gram = 0
			for gram in trigrams:
				if(counter_gram%3 == 0):
					tri_seq_list_0.append(gram)
				if(counter_gram%3 == 1):
					tri_seq_list_1.append(gram)
				if(counter_gram%3 == 2):
					tri_seq_list_2.append(gram)
				counter_gram += 1

			tri_seq_str_0 = ""
			tri_seq_str_1 = ""
			tri_seq_str_2 = ""

			for gram in tri_seq_list_0:
				tri_seq_str_0 += gram[0] + gram[1] + gram[2] + " "
			for gram in tri_seq_list_1:
				tri_seq_str_1 += gram[0] + gram[1] + gram[2] + " "
			for gram in tri_seq_list_2:
				tri_seq_str_2 += gram[0] + gram[1] + gram[2] + " "
			
			tri_seq_tot_0 += tri_seq_str_0 + "end end end end end end "		
			tri_seq_tot_1 += tri_seq_str_1 + "end end end end end end "		
			tri_seq_tot_2 += tri_seq_str_2 + "end end end end end end "		

		# print(tri_seq_tot_0, "\n\n", tri_seq_tot_1, "\n\n", tri_seq_tot_2, "\n\n")	

		tri_seq_tot = tri_seq_tot_0
		tri_seq_tot += tri_seq_tot_1
		tri_seq_tot += tri_seq_tot_2
		with open(file_path, "w") as otput_file:
			otput_file.write(tri_seq_tot)

def seq_to_vec_mini_batches(min_no_of_seq = 200):
	file_path = './../data/db_' + str(min_no_of_seq) +'_pickle'
	file_ip = open(file_path, 'rb')
	familywise_db = pickle.load(file_ip)
	file_ip.close()

	# removing families with sequence length greater then 1000
	print(len(familywise_db.keys()))
	del_fams = []
	for fam in familywise_db.keys():
		fam_seq = familywise_db[fam]
		for seq_no in range(len(fam_seq)):
			data = fam_seq[seq_no]
			seq = data[1]
			if(len(seq) > 1000):
				del_fams.append(fam)
				break
	for fam in del_fams:
		del familywise_db[fam]
	print(len(familywise_db.keys()))
	
	# create family - number mapping
	fam_num = {}
	fams = list(familywise_db.keys())
	for i in range(len(fams)):
		fam_num[fams[i]] = i

	file_path = './../data/families_map_filtered' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(fam_num, output)
		output.close()
	
	# 200*0.70 = 140 train
	# 200*0.15 =  30 test
	# 200*0.15 =  30 cv

	data_train = {}
	data_test = {}
	data_cv = {}
	data_test_list = []
	data_cv_list = []
	word_to_glove = read_glove_vec_files()

	# Create data_train, data_test, data_cv
	# It will be a dictonary with
	# number to batch mapping

	for fam in fams:
		data = familywise_db[fam]
		data_len = len(data)
		family_number = fam_num[fam]
		# print("Before deleting", len(data), 3 * data_len // 10)
		for i in range(3 * data_len // 10):
			rec = data[0]
			glove_of_seq = seq_to_glove_vector(rec[1], word_to_glove)
			temp = [glove_of_seq, family_number]
			if( i < 15 * data_len // 100):
				data_test_list.append(temp)
			else :
				data_cv_list.append(temp)
			del data[0]

		# print("After deleting", len(data))
		familywise_db[fam] = data

	# print("Length of data_cv, data_test : ", len(data_cv_list), len(data_test_list))
	# 35746 35492

	no_of_batches_data_test = len(data_test_list) // 1000
	for i in range(len(data_test_list) // 1000):
		data = data_test_list[i*1000: i*1000+1000]
		inp = []
		op = []
		freq = []
		for j in range(len(data)):
			curr_rec = data[j]
			inp.append(curr_rec[0])
			op.append(curr_rec[1])
			freq.append(1)
			# freq wont matter in case of cv and test
		temp = [inp, op, freq]
		data_test[i] = temp
		# print("Batch no : ", i, len(temp[0]))
	data = data_test_list[no_of_batches_data_test*1000:]
	inp = []
	op = []
	freq = []
	for j in range(len(data)):
		curr_rec = data[j]
		inp.append(curr_rec[0])
		op.append(curr_rec[1])
		freq.append(1)
	temp = [inp, op, freq]
	data_test[no_of_batches_data_test] = temp

	# count = 0
	# for k in data_test.keys():
	# 	data = data_test[k]
	# 	print(k, len(data[0]))
	# 	count += len(data[0])
	# print(count)
	# 0 - 35 batches
	# 35492

	file_path = './../data/data_test_glove_filtered_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(data_test, output)
		output.close()
	
	no_of_batches_data_cv = len(data_cv_list) // 1000
	for i in range(len(data_cv_list) // 1000):
		data = data_cv_list[i*1000: i*1000+1000]
		inp = []
		op = []
		freq = []
		for j in range(len(data)):
			curr_rec = data[j]
			inp.append(curr_rec[0])
			op.append(curr_rec[1])
			freq.append(1)
			# freq wont matter in case of cv and test
		temp = [inp, op, freq]
		data_cv[i] = temp
	data = data_cv_list[no_of_batches_data_cv*1000:]
	inp = []
	op = []
	freq = []
	for j in range(len(data)):
		curr_rec = data[j]
		inp.append(curr_rec[0])
		op.append(curr_rec[1])
		freq.append(1)
	temp = [inp, op, freq]
	data_cv[no_of_batches_data_cv] = temp

	# count = 0
	# for k in data_cv.keys():
	# 	data = data_cv[k]
	# 	print(k, len(data[0]))
	# 	count += len(data[0])
	# print(count)
	# 0 - 35 batches 
	# 35746 examples in total

	file_path = './../data/data_cv_glove_filtered_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(data_cv, output)
		output.close()
	
	# considering 2 examples per class 
	# we can have 70 balanced batches with 498 examples
	# and rest of the batches would be unbalanced
	# number the balnced batches 1-70
	# other batches get numbered accordingly
	# when train call batches in reverse order

	# create class to number mapping and also class
	# frequency in the file

	fam_freq = {}
	for fam in fams:
		data = familywise_db[fam]
		fam_freq[fam_num[fam]] = len(data)
	# print("Minimum : ", ans) 140

	for i in range(70):
		inp = []	
		op = []	
		freq = []
		for fam in familywise_db.keys():
			data = familywise_db[fam]
			family_number = fam_num[fam]
			for j in range(2):
				rec = data[j]
				inp.append(seq_to_glove_vector(rec[1], word_to_glove))				
				op.append(family_number)
				freq.append(fam_freq[family_number])
			del data[0]
			del data[0]
			familywise_db[fam] = data
		temp = [inp, op, freq]
		data_train[i] = temp

	# total = 0
	# for i in range(70):
	# 	total += len(data_train[i][0])
	# 	print(i, len(data_train[i][0]), Counter(data_train[i][1]))
	# print(total)
	# 69720
	# 2 in all batches for all elements

	data_train_list = []
	fams = list(familywise_db.keys())

	while not (len(fams) == 0) :
		for fam in fams:
			data = familywise_db[fam]
			if(len(data) == 0):
				fams.remove(fam)
				continue
			rec = data[0]
			glove_of_seq = seq_to_glove_vector(rec[1], word_to_glove)
			family_number = fam_num[fam]
			temp = [glove_of_seq, family_number]
			data_train_list.append(temp)
			del data[0]
			
	# print(len(data_train_list))
	# 97212
	# 97212 + 69720 = 166932 (matches)

	no_of_batches_data_train = len(data_train_list) // 1000
	for i in range(len(data_train_list) // 1000):
		data = data_train_list[i*1000: i*1000+1000]
		inp = []
		op = []
		freq = []
		for j in range(len(data)):
			curr_rec = data[j]
			inp.append(curr_rec[0])
			op.append(curr_rec[1])
			freq.append(fam_freq[curr_rec[1]])
		temp = [inp, op, freq]
		data_train[i + 70] = temp
	data = data_train_list[no_of_batches_data_train*1000:]
	inp = []
	op = []
	freq = []
	for j in range(len(data)):
		curr_rec = data[j]
		inp.append(curr_rec[0])
		op.append(curr_rec[1])
		freq.append(fam_freq[curr_rec[1]])
	temp = [inp, op, freq]
	data_train[no_of_batches_data_train + 70] = temp

	# debug = input()
	# count = 0
	# for k in data_train.keys():
	# 	data = data_train[k]
	# 	print(k, len(data[0]), Counter(data[1]))
	# 	count += len(data[0])
	# print(count) 166932
	# 167 batches with 998 examples in first 
	# 70 batches (batch no 0-69)
	# and 1000 examples in last but one batch
	# with 212 examples in last batch belonging to class 141

	file_path = './../data/data_train_glove_filtered_' + str(min_no_of_seq) +'_pickle'
	if(not os.path.isfile(file_path)):
		output = open(file_path, 'ab')
		pickle.dump(data_train, output)
		output.close()

def padding_all_batches(min_no_of_seq = 200):
	file_path = './../data/data_train_glove_filtered_' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_train = pickle.load(file_ip)
	file_ip.close()

	file_path = './../data/data_cv_glove_filtered_' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_cv = pickle.load(file_ip)
	file_ip.close()

	file_path = './../data/data_test_glove_filtered_' + str(min_no_of_seq) + '_pickle'
	file_ip = open(file_path, 'rb')
	data_test = pickle.load(file_ip)
	file_ip.close()

	def func(data_train, name_file):
		no_of_batches = len(data_train.keys())
		for batch_no in range(no_of_batches-1, -1, -1):
			print("File anem and iteration : ", name_file, batch_no)
			data_batch = data_train[batch_no]
			batch_size = len(data_batch[1])
			x = data_batch[0]
			y = data_batch[1]
			freq = data_batch[2]
			max_length = 0
			seq_length = []
			for data in x:
				seq_length.append(len(data))
				max_length = max(max_length, len(data))
			x_n = [ row + (zero_100_list)*(max_length-len(row)) for row in x]
			x_padded = np.array(x_n)
			data_train[batch_no]  = [ x_n, y, freq, seq_length]
		save_obj(data_train,'./../data/'+ name_file + '_filt_pad_'+ str(min_no_of_seq) +'_pkl')

	func(data_train, 'data_train')
	func(data_test, 'data_test')
	func(data_cv, 'data_cv')

if __name__=="__main__":
	# Ran these once, so files are saved 
	seq_merger_one_file()
	# seq_to_vec_mini_batches(200)
	# padding_all_batches(200)
	# save_familywise_db()
	# save_familywise_db(100)
	# save_familywise_db(50)
	# seq_to_glove_vector("QMGLAE")
	# read_glove_vec_files()
	# map_creator()

	# new_list = ['1', '2', '3', '4', '5', '6', '7', '8']
	# print(new_list)
	# for i in range(1, 5):
	# 	del new_list[1]
	# print(new_list)
	# word_to_glove = read_glove_vec_files()
	# print(seq_to_glove_vector('QMGL', word_to_glove))


"""
To remove sequences with length > 1000, 1% data will be missed.
"""




	