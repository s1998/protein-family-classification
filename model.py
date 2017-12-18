#import the necessary libraies
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from random import shuffle
import numpy as np
from sklearn.metrics import classification_report as c_metric

# get train, test and validate data
def get_data_glove(min_no_of_seq = 200):
	file_path = './data/data_train_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_train = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_cv_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_cv = pickle.load(file_ip)
	file_ip.close()

	file_path = './data/data_test_filt_pad_'+ str(min_no_of_seq) +'_pkl'
	file_ip = open(file_path, 'rb')
	data_test = pickle.load(file_ip)
	file_ip.close()

	return data_train, data_test, data_cv

# defining the class for our model

# model is an object of this class

# the initializer makes the data computation graph

# the functions of the class help in running 
# different components of the computation graph 

class RnnForPfcModel:
	def __init__(self, 
		num_classes = 549, 
		hidden_units=100, 
		learning_rate=0.001):

		# defining placeholders for input
		self.seq_length = tf.placeholder(tf.int64, [None])
		self.freq = tf.placeholder(tf.float64, [None])
		self.x_input = tf.placeholder(tf.float64, [None, None, 100], name = 'x_ip')
		self.y_input = tf.placeholder(tf.int64, [None], name = 'y_ip')
		
		# freq_inv is (1 / class frequency) to deal with class imbalance 
		self.freq_inv = tf.div(self.freq * 0 + 1, self.freq)
		self.learning_rate = learning_rate	

		# convert input to one hot representation
		self.y_input_o = tf.one_hot(indices = self.y_input, 
									depth = num_classes,
									on_value = 1.0,
									off_value = 0.0,
									axis = -1)
		
		# helps in xavier initialization
		self.hidden_units = tf.constant(hidden_units, dtype = tf.float64)
		
		# define weights and biases
		self.weights_f = tf.Variable(tf.random_uniform(shape=[hidden_units, hidden_units], maxval=1, dtype=tf.float64) / (tf.sqrt(self.hidden_units / 2)), dtype=tf.float64)
		self.weights_p = tf.Variable(tf.random_uniform(shape=[hidden_units, hidden_units], maxval=1, dtype=tf.float64) / (tf.sqrt(self.hidden_units / 2)), dtype=tf.float64)
		self.weights_h = tf.Variable(tf.random_uniform(shape=[hidden_units, num_classes],  maxval=1, dtype=tf.float64) / (tf.sqrt(self.hidden_units)), dtype=tf.float64)
		self.biases_f = tf.Variable(tf.zeros(shape=[hidden_units], dtype=tf.float64), dtype=tf.float64) + 0.01
		self.biases_p = tf.Variable(tf.zeros(shape=[hidden_units], dtype=tf.float64), dtype=tf.float64) + 0.01
		self.biases_h = tf.Variable(tf.zeros(shape=[num_classes], dtype=tf.float64), dtype=tf.float64) 
		
		# define rnn forward cell
		self.rnn_fcell = rnn.BasicLSTMCell(num_units = hidden_units, 
										   forget_bias = 1.0,
										   activation = tf.tanh)
		
		# create dynamic rnn for efficient calculation when using
		# data that contains sequences with variable length
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_fcell,
													  self.x_input,
													  sequence_length = self.seq_length,
													  dtype = tf.float64)
		
		# get maxpooling over the outputs
		self.outputs_maxpooled = tf.reduce_max(self.outputs, axis = 1)
		
		# reshape the outputs
		self.outputs_f = tf.reshape(self.outputs[:, -1, :], [-1, hidden_units])
		self.outputs_p = tf.reshape(self.outputs_maxpooled, [-1, hidden_units])
		
		# adding next layer in the compuatation graph
		self.h_predicted = (tf.matmul(self.outputs_f, self.weights_f) + self.biases_f 
   						   + tf.matmul(self.outputs_p, self.weights_p) + self.biases_p)
		
		# predicting the outputs
		self.y_predicted = tf.matmul(tf.nn.relu(self.h_predicted), self.weights_h) + self.biases_h

		# calculating the unweighted loss
		self.loss_unweighted = (
					tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input_o))
		# calculate the weighted loss (multiply by inverse of class frequency)
		self.loss_weighted = tf.multiply(self.loss_unweighted, self.freq_inv)
		# calculate the mean weighted loss
		self.loss_reduced = tf.reduce_mean(self.loss_weighted) 

		# define optimizer and trainers
		self.optimizer_1 = tf.train.AdamOptimizer(learning_rate = 0.01)
		self.trainer_1 = self.optimizer_1.minimize(self.loss_reduced)

		self.optimizer_2 = tf.train.AdamOptimizer(learning_rate = 0.001)
		self.trainer_2 = self.optimizer_2.minimize(self.loss_reduced)

		self.optimizer_3 = tf.train.AdamOptimizer(learning_rate = 0.0001)
		self.trainer_3 = self.optimizer_3.minimize(self.loss_reduced)

		self.optimizer_4 = tf.train.AdamOptimizer(learning_rate = 0.00001)
		self.trainer_4 = self.optimizer_4.minimize(self.loss_reduced)

		self.optimizer_5 = tf.train.AdamOptimizer(learning_rate = 0.000001)
		self.trainer_5 = self.optimizer_5.minimize(self.loss_reduced)
		
		self.optimizer_6 = tf.train.AdamOptimizer(learning_rate = 0.0000001)
		self.trainer_6 = self.optimizer_6.minimize(self.loss_reduced)
		
		# creating a session variable to run different compenents
    	# of the computation graph and initialize the variables
		self.sess = tf.Session()
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)

		# get the accuracy
		self.get_equal = tf.equal(tf.argmax(self.y_input_o, 1), tf.argmax(self.y_predicted, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float64))

	# get the predictions for current data
	def predict(self, x, y, seq_length, freq):
		result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})
		result = np.argmax(result, axis=1)
		result = np.reshape(result, [-1])
		return result
	
	# optimizer_i calls trainer_i
	# variable learning rate helps in converging faster
	# initiallly we keep high learning rate and then decrease
	# the learning rate as accuracy increases.

	def optimize_1(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_1, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_2(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_2, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_3(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_3, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_4(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_4, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def optimize_5(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_5, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})
	
	def optimize_6(self, x, y, seq_length, freq):
		result = self.sess.run(self.trainer_6, feed_dict={self.x_input: x, self.y_input: y, self.seq_length:seq_length, self.freq:freq})

	def cross_validate(self, x, y, seq_length, freq):
		result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length, self.freq:freq})
		return result

	# get the loss for current data
	def get_loss(self, x, y, seq_length, freq):
		result = self.sess.run(self.loss_reduced, feed_dict={self.x_input:x, self.y_input:y, self.seq_length:seq_length, self.freq:freq})
		return result

# intially trainer_1 with highest learning rate will be called
use_optimizer = 1

def train_on_train_data(epoch, model, data_train, data_test):
	global use_optimizer
	no_of_batches = len(data_train.keys())
	for batch_no in range(70):
		print("Iteration number, batch number : ", epoch, batch_no)
		
		# get the input data
		data_batch = data_train[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = [140] * batch_size
		seq_length = data_batch[3]
		x = np.array(x)
		y_n = np.array(y)
		freq = np.array(freq)

		# call the trainer from computation graph
		if use_optimizer == 1:
			model.optimize_1(x, y, seq_length, freq)
		elif use_optimizer == 2:
			model.optimize_2(x, y, seq_length, freq)
		elif use_optimizer == 3:
			model.optimize_3(x, y, seq_length, freq)
		elif use_optimizer == 4:
			model.optimize_4(x, y, seq_length, freq)
		elif use_optimizer == 5:
			model.optimize_5(x, y, seq_length, freq)
		elif use_optimizer == 6:
			model.optimize_6(x, y, seq_length, freq)

		# get accuracy and predicted values of y
		accuracy = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted = y_predicted_
		print("Training data accuracy : ", accuracy)
		print("Training data loss     : ", model.get_loss(x, y, seq_length, freq))
		
		# change the learning rate as required 
		# (the poins of change have been obtained experimentally i.e. after 
		# running the code multiple times)
		if(use_optimizer == 1 and accuracy > 0.80):
			use_optimizer = 2
		if(use_optimizer == 2 and accuracy > 0.85):
			use_optimizer = 3
		if(use_optimizer == 3 and accuracy > 0.90):
			use_optimizer = 4
		# gave good results with 93% accuracy, 0.83 F1 score
		# if(use_optimizer == 4 and accuracy > 0.92):
		# 	use_optimizer = 5
		# if(use_optimizer == 4 and accuracy > 0.93):
		# 	use_optimizer = 6

# The functions res_on_train_data, res_on_test_data, res_on_cv_data are
# all similar, so we have commented res_on_train_data.
def res_on_train_data(model, data_train):
	no_of_batches = len(data_train.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
		print("Iteration number, batch number : ", epoch, batch_no)
		
		# get input data
		data_batch = data_train[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = [140] * batch_size
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)

		# get accuracy, predictions
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	
	# get final accuracy and predictions
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on train data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

def res_on_test_data(model, data_test):
	no_of_batches = len(data_test.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_test[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = data_batch[2]
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on test data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

def res_on_cv_data(model, data_cv):
	no_of_batches = len(data_cv.keys())
	y_predicted = []
	y_actual = []
	correct = 0
	total_samples = 0
	for batch_no in range(no_of_batches-1, -1, -1):
		print("Iteration number, batch number : ", epoch, batch_no)
		data_batch = data_cv[batch_no]
		batch_size = len(data_batch[1])
		x = data_batch[0]
		y = data_batch[1]
		freq = data_batch[2]
		seq_length = data_batch[3]
		x = np.array(x)
		y_actual.extend(y)
		y = np.array(y)
		freq = np.array(freq)
		accuracy_kmown = model.cross_validate(x, y, seq_length, freq)
		y_predicted_ = model.predict(x, y, seq_length, freq)
		y_predicted.extend(y_predicted_)
		correct += accuracy_kmown * batch_size
		total_samples += batch_size
	accuracy = (correct * 100) / (total_samples)
	correct_1 = len([i for i,j in zip(y_actual, y_predicted) if i == j])
	print("Accuracy on cv data : ", accuracy, correct, correct_1)
	print("Lengths of y_actual and y_predicted : ", len(y_actual), len(y_predicted))
	print (c_metric(y_actual, y_predicted))

if __name__=="__main__":
	n_epochs = 50
	batch_size = 1000
	hidden_units = 100
	num_classes = 498
	data_train, data_test, data_cv = get_data_glove(200)
	model = RnnForPfcModel(num_classes = 498, hidden_units=100, learning_rate=0.001)
	saver = tf.train.Saver()
	for epoch in range(n_epochs):
		train_on_train_data(epoch, model, data_train, data_test)
		# res_on_train_data(model, data_train) [calling this takes lot of time, also not needed]
		res_on_test_data(model, data_test)
		res_on_cv_data(model, data_cv)
		saver.save(model.sess, './data/ckpts/' + 'model.ckpt', global_step= epoch)

