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

class RnnForPfcModelSix:
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
	n_epochs = 10
	batch_size = 1000
	hidden_units = 100
	num_classes = 498
	data_train, data_test, data_cv = get_data_glove(200)
	model = RnnForPfcModelSix(num_classes = 498, hidden_units=100, learning_rate=0.001)
	saver = tf.train.Saver()
	for epoch in range(n_epochs):
		train_on_train_data(epoch, model, data_train, data_test)
		# res_on_train_data(model, data_train) [calling this takes lot of time, also not needed]
		res_on_test_data(model, data_test)
		res_on_cv_data(model, data_cv)
		saver.save(model.sess, './data/ckpts/' + 'model.ckpt', global_step= epoch)

"""
Iteration number, batch number :  3 0
Training data accuracy :  0.819277108434
Training data loss     :  0.00597934742096
Iteration number, batch number :  3 1
Training data accuracy :  0.809236947791
Training data loss     :  0.00604017527044
Iteration number, batch number :  3 2
Training data accuracy :  0.815261044177
Training data loss     :  0.00594495014262
Iteration number, batch number :  3 3
Training data accuracy :  0.820281124498
Training data loss     :  0.00592881642384
Iteration number, batch number :  3 4
Training data accuracy :  0.826305220884
Training data loss     :  0.00579686092087
Iteration number, batch number :  3 5
Training data accuracy :  0.848393574297
Training data loss     :  0.00556887689244
Iteration number, batch number :  3 6
Training data accuracy :  0.830321285141
Training data loss     :  0.00596083975579
Iteration number, batch number :  3 7
Training data accuracy :  0.838353413655
Training data loss     :  0.00508792306155
Iteration number, batch number :  3 8
Training data accuracy :  0.822289156627
Training data loss     :  0.00547598246311
Iteration number, batch number :  3 9
Training data accuracy :  0.833333333333
Training data loss     :  0.00548689275186
Iteration number, batch number :  3 10
Training data accuracy :  0.853413654618
Training data loss     :  0.00494284609889
Iteration number, batch number :  3 11
Training data accuracy :  0.827309236948
Training data loss     :  0.00540464753374
Iteration number, batch number :  3 12
Training data accuracy :  0.839357429719
Training data loss     :  0.00538859067519
Iteration number, batch number :  3 13
Training data accuracy :  0.823293172691
Training data loss     :  0.00577323696197
Iteration number, batch number :  3 14
Training data accuracy :  0.836345381526
Training data loss     :  0.00513101506029
Iteration number, batch number :  3 15
Training data accuracy :  0.848393574297
Training data loss     :  0.00506499185601
Iteration number, batch number :  3 16
Training data accuracy :  0.854417670683
Training data loss     :  0.00473358341941
Iteration number, batch number :  3 17
Training data accuracy :  0.831325301205
Training data loss     :  0.00502986929377
Iteration number, batch number :  3 18
Training data accuracy :  0.828313253012
Training data loss     :  0.00545341663542
Iteration number, batch number :  3 19
Training data accuracy :  0.850401606426
Training data loss     :  0.00523235135789
Iteration number, batch number :  3 20
Training data accuracy :  0.860441767068
Training data loss     :  0.00490600422409
Iteration number, batch number :  3 21
Training data accuracy :  0.834337349398
Training data loss     :  0.00503039446787
Iteration number, batch number :  3 22
Training data accuracy :  0.846385542169
Training data loss     :  0.0047607024808
Iteration number, batch number :  3 23
Training data accuracy :  0.847389558233
Training data loss     :  0.00477209111393
Iteration number, batch number :  3 24
Training data accuracy :  0.842369477912
Training data loss     :  0.00517890656622
Iteration number, batch number :  3 25
Training data accuracy :  0.848393574297
Training data loss     :  0.00523880514286
Iteration number, batch number :  3 26
Training data accuracy :  0.859437751004
Training data loss     :  0.00495883164326
Iteration number, batch number :  3 27
Training data accuracy :  0.841365461847
Training data loss     :  0.0050470377363
Iteration number, batch number :  3 28
Training data accuracy :  0.867469879518
Training data loss     :  0.00461717862984
Iteration number, batch number :  3 29
Training data accuracy :  0.836345381526
Training data loss     :  0.00510340641663
Iteration number, batch number :  3 30
Training data accuracy :  0.846385542169
Training data loss     :  0.00495002155457
Iteration number, batch number :  3 31
Training data accuracy :  0.838353413655
Training data loss     :  0.00520936057841
Iteration number, batch number :  3 32
Training data accuracy :  0.850401606426
Training data loss     :  0.00509798614009
Iteration number, batch number :  3 33
Training data accuracy :  0.877510040161
Training data loss     :  0.00466967688021
Iteration number, batch number :  3 34
Training data accuracy :  0.867469879518
Training data loss     :  0.0045362744703
Iteration number, batch number :  3 35
Training data accuracy :  0.859437751004
Training data loss     :  0.00441098016938
Iteration number, batch number :  3 36
Training data accuracy :  0.850401606426
Training data loss     :  0.00461525868364
Iteration number, batch number :  3 37
Training data accuracy :  0.871485943775
Training data loss     :  0.00467494422607
Iteration number, batch number :  3 38
Training data accuracy :  0.864457831325
Training data loss     :  0.00470697695461
Iteration number, batch number :  3 39
Training data accuracy :  0.88453815261
Training data loss     :  0.00411484799714
Iteration number, batch number :  3 40
Training data accuracy :  0.850401606426
Training data loss     :  0.00502973397234
Iteration number, batch number :  3 41
Training data accuracy :  0.855421686747
Training data loss     :  0.00462641138445
Iteration number, batch number :  3 42
Training data accuracy :  0.86546184739
Training data loss     :  0.004592698401
Iteration number, batch number :  3 43
Training data accuracy :  0.86546184739
Training data loss     :  0.0045888477722
Iteration number, batch number :  3 44
Training data accuracy :  0.866465863454
Training data loss     :  0.00421606248674
Iteration number, batch number :  3 45
Training data accuracy :  0.868473895582
Training data loss     :  0.00440673499137
Iteration number, batch number :  3 46
Training data accuracy :  0.867469879518
Training data loss     :  0.00440146056229
Iteration number, batch number :  3 47
Training data accuracy :  0.867469879518
Training data loss     :  0.00474293285651
Iteration number, batch number :  3 48
Training data accuracy :  0.854417670683
Training data loss     :  0.00493186939269
Iteration number, batch number :  3 49
Training data accuracy :  0.849397590361
Training data loss     :  0.0046158809434
Iteration number, batch number :  3 50
Training data accuracy :  0.862449799197
Training data loss     :  0.00465136049761
Iteration number, batch number :  3 51
Training data accuracy :  0.836345381526
Training data loss     :  0.00508176848581
Iteration number, batch number :  3 52
Training data accuracy :  0.843373493976
Training data loss     :  0.00555385611994
Iteration number, batch number :  3 53
Training data accuracy :  0.841365461847
Training data loss     :  0.00529098922355
Iteration number, batch number :  3 54
Training data accuracy :  0.847389558233
Training data loss     :  0.00524054800629
Iteration number, batch number :  3 55
Training data accuracy :  0.836345381526
Training data loss     :  0.00505897180763
Iteration number, batch number :  3 56
Training data accuracy :  0.84437751004
Training data loss     :  0.00527979039576
Iteration number, batch number :  3 57
Training data accuracy :  0.833333333333
Training data loss     :  0.00538221068953
Iteration number, batch number :  3 58
Training data accuracy :  0.824297188755
Training data loss     :  0.00538676705542
Iteration number, batch number :  3 59
Training data accuracy :  0.838353413655
Training data loss     :  0.00562819932135
Iteration number, batch number :  3 60
Training data accuracy :  0.838353413655
Training data loss     :  0.00531861853046
Iteration number, batch number :  3 61
Training data accuracy :  0.826305220884
Training data loss     :  0.00545997473354
Iteration number, batch number :  3 62
Training data accuracy :  0.836345381526
Training data loss     :  0.00540625051405
Iteration number, batch number :  3 63
Training data accuracy :  0.825301204819
Training data loss     :  0.00555137552588
Iteration number, batch number :  3 64
Training data accuracy :  0.842369477912
Training data loss     :  0.00577942000103
Iteration number, batch number :  3 65
Training data accuracy :  0.831325301205
Training data loss     :  0.00595717665071
Iteration number, batch number :  3 66
Training data accuracy :  0.83734939759
Training data loss     :  0.00530889594171
Iteration number, batch number :  3 67
Training data accuracy :  0.852409638554
Training data loss     :  0.0054386882694
Iteration number, batch number :  3 68
Training data accuracy :  0.835341365462
Training data loss     :  0.00567919465119
Iteration number, batch number :  3 69
Training data accuracy :  0.830321285141
Training data loss     :  0.00539723457392
Iteration number, batch number :  3 35
Iteration number, batch number :  3 34
Iteration number, batch number :  3 33
Iteration number, batch number :  3 32
Iteration number, batch number :  3 31
Iteration number, batch number :  3 30
Iteration number, batch number :  3 29
Iteration number, batch number :  3 28
Iteration number, batch number :  3 27
Iteration number, batch number :  3 26
Iteration number, batch number :  3 25
Iteration number, batch number :  3 24
Iteration number, batch number :  3 23
Iteration number, batch number :  3 22
Iteration number, batch number :  3 21
Iteration number, batch number :  3 20
Iteration number, batch number :  3 19
Iteration number, batch number :  3 18
Iteration number, batch number :  3 17
Iteration number, batch number :  3 16
Iteration number, batch number :  3 15
Iteration number, batch number :  3 14
Iteration number, batch number :  3 13
Iteration number, batch number :  3 12
Iteration number, batch number :  3 11
Iteration number, batch number :  3 10
Iteration number, batch number :  3 9
Iteration number, batch number :  3 8
Iteration number, batch number :  3 7
Iteration number, batch number :  3 6
Iteration number, batch number :  3 5
Iteration number, batch number :  3 4
Iteration number, batch number :  3 3
Iteration number, batch number :  3 2
Iteration number, batch number :  3 1
Iteration number, batch number :  3 0
Accuracy on test data :  75.8734362673 26929.0 26929
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.79      0.91      0.85        45
          1       0.70      0.95      0.80        41
          2       0.64      0.81      0.71        37
          3       0.85      0.96      0.90        52
          4       0.96      0.54      0.69       130
          5       0.92      0.97      0.95        36
          6       0.46      0.63      0.53        51
          7       0.70      0.60      0.64        47
          8       0.41      0.38      0.40        76
          9       0.96      1.00      0.98        51
         10       0.95      0.93      0.94        45
         11       0.37      0.45      0.41        31
         12       0.94      0.86      0.90        79
         13       0.70      0.84      0.76        85
         14       0.90      0.85      0.87       143
         15       0.51      0.39      0.44       120
         16       0.91      0.94      0.93        34
         17       0.65      0.31      0.42       124
         18       0.89      0.77      0.82        70
         19       0.50      0.56      0.53        91
         20       0.40      0.62      0.48        39
         21       0.90      0.34      0.49        53
         22       0.53      0.51      0.52        45
         23       0.48      0.73      0.58        33
         24       0.80      0.80      0.80        40
         25       0.98      0.98      0.98        85
         26       0.92      0.73      0.81       132
         27       0.93      0.86      0.89        90
         28       0.34      0.53      0.42        30
         29       0.85      0.84      0.85        96
         30       0.89      0.86      0.88        74
         31       0.97      0.86      0.91        79
         32       1.00      0.98      0.99       269
         33       0.97      0.90      0.93        31
         34       0.44      0.72      0.55        32
         35       0.88      0.95      0.91        77
         36       0.88      0.91      0.90        66
         37       0.74      0.82      0.78       114
         38       0.06      0.01      0.01       121
         39       0.88      0.82      0.85        34
         40       0.31      0.45      0.37        76
         41       0.91      0.64      0.75        92
         42       0.64      0.72      0.68        58
         43       0.95      0.89      0.92        63
         44       0.60      0.80      0.69        30
         45       0.83      0.95      0.88        40
         46       0.00      0.00      0.00       161
         47       0.58      0.77      0.66        47
         48       0.89      0.94      0.92        36
         49       0.78      0.95      0.85        37
         50       0.78      0.65      0.71       123
         51       0.60      0.49      0.54       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.95      0.90      0.93        70
         55       0.98      0.93      0.95        94
         56       0.59      0.61      0.60        66
         57       0.70      0.63      0.67        79
         58       0.97      0.92      0.94        98
         59       0.95      0.89      0.92        92
         60       1.00      0.88      0.94        33
         61       0.84      0.93      0.88        41
         62       0.00      0.00      0.00        30
         63       0.83      0.88      0.86        74
         64       0.84      1.00      0.92        38
         65       0.95      0.88      0.92        95
         66       0.76      0.66      0.70        94
         67       0.86      0.90      0.88       132
         68       0.39      0.43      0.41        72
         69       0.36      0.53      0.43        30
         70       0.69      0.79      0.73        42
         71       0.61      0.80      0.69        44
         72       0.33      0.64      0.44        42
         73       0.86      0.84      0.85        37
         74       0.64      0.89      0.74        98
         75       0.94      0.87      0.90       163
         76       0.91      0.89      0.90        65
         77       0.97      0.94      0.96        70
         78       0.90      0.85      0.88        33
         79       0.61      0.76      0.68        33
         80       0.78      0.69      0.73        71
         81       0.96      0.98      0.97       131
         82       0.80      0.72      0.76       174
         83       0.67      0.74      0.71        47
         84       0.21      0.23      0.22        44
         85       0.95      0.93      0.94       110
         86       0.54      0.46      0.50        97
         87       0.88      0.77      0.82        83
         88       0.71      0.84      0.77        43
         89       0.71      0.91      0.80        35
         90       0.72      0.82      0.77        38
         91       0.67      0.61      0.64        51
         92       0.42      0.68      0.52        40
         93       0.96      0.92      0.94        48
         94       0.91      0.89      0.90        71
         95       0.79      0.79      0.79       117
         96       0.53      0.84      0.65        38
         97       0.69      0.90      0.78        30
         98       0.88      0.86      0.87        98
         99       0.64      0.72      0.68        69
        100       0.92      0.91      0.91        98
        101       0.68      0.64      0.65        85
        102       0.36      0.54      0.43        56
        103       0.97      0.92      0.94        36
        104       0.65      0.60      0.62        40
        105       0.68      0.55      0.61        65
        106       0.59      0.69      0.64        32
        107       1.00      0.59      0.74        34
        108       0.66      0.75      0.71        81
        109       0.62      0.26      0.37        77
        110       0.97      0.91      0.94       121
        111       0.52      0.53      0.52        53
        112       0.90      0.89      0.89        88
        113       0.76      0.97      0.85        30
        114       0.84      0.88      0.86       127
        115       0.93      0.72      0.81        78
        116       0.57      0.53      0.55       108
        117       0.93      0.82      0.87       110
        118       0.49      0.40      0.44        88
        119       0.89      0.79      0.84       106
        120       0.59      0.77      0.67        69
        121       0.78      0.82      0.80        39
        122       0.84      0.77      0.80       124
        123       0.83      0.81      0.82        59
        124       1.00      0.96      0.98        69
        125       0.65      0.87      0.74        38
        126       0.74      0.61      0.67       113
        127       0.98      1.00      0.99        55
        128       0.72      0.84      0.78        82
        129       0.93      0.90      0.91       125
        130       0.57      0.85      0.68        47
        131       0.85      0.36      0.50       112
        132       0.98      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.45      0.56      0.50        32
        135       0.82      0.79      0.80        80
        136       0.74      0.69      0.71       103
        137       0.75      0.79      0.77       134
        138       0.95      0.91      0.93        88
        139       0.17      0.11      0.13        37
        140       0.93      0.98      0.96        44
        141       0.85      0.90      0.87        61
        142       0.99      0.94      0.97       103
        143       0.51      0.67      0.58        51
        144       0.48      0.65      0.55        72
        145       0.73      0.98      0.83        49
        146       0.70      0.75      0.73        85
        147       0.89      0.87      0.88        93
        148       0.88      0.86      0.87        57
        149       0.83      0.86      0.85       109
        150       0.89      0.64      0.75        76
        151       0.65      0.69      0.67        75
        152       0.91      0.91      0.91        54
        153       0.49      0.78      0.60        32
        154       0.83      0.88      0.85        73
        155       0.80      0.59      0.68       189
        156       0.63      0.46      0.53        92
        157       0.94      0.94      0.94        31
        158       0.45      0.14      0.21       100
        159       0.48      0.42      0.45       108
        160       0.25      0.44      0.31        39
        161       0.63      0.75      0.69        57
        162       0.94      1.00      0.97        33
        163       0.38      0.47      0.42        70
        164       0.95      0.80      0.87       159
        165       0.50      0.72      0.59        39
        166       0.89      0.94      0.92        71
        167       0.85      0.80      0.82        64
        168       0.90      0.81      0.85        75
        169       0.51      0.64      0.57        39
        170       0.99      0.94      0.96       140
        171       0.15      0.08      0.11        61
        172       0.82      0.90      0.86        88
        173       0.63      0.69      0.66        39
        174       0.82      0.89      0.86        37
        175       0.73      0.80      0.77        45
        176       0.32      0.50      0.39        42
        177       0.57      0.84      0.68        31
        178       0.93      0.91      0.92        55
        179       0.37      0.55      0.44        31
        180       0.61      0.62      0.61        82
        181       0.57      0.65      0.61        31
        182       0.66      0.66      0.66        87
        183       0.70      0.85      0.77        41
        184       0.85      0.93      0.89        57
        185       0.76      0.87      0.81        30
        186       1.00      0.95      0.97        56
        187       0.71      1.00      0.83        30
        188       0.97      1.00      0.99        33
        189       0.67      0.80      0.73        46
        190       0.91      0.99      0.95       107
        191       0.90      0.95      0.93        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.78      0.74      0.76       152
        195       0.95      0.85      0.90        71
        196       0.92      0.95      0.94        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.38      0.73      0.50        37
        200       0.94      1.00      0.97        34
        201       0.17      0.27      0.21        30
        202       0.74      0.81      0.77        53
        203       0.89      0.97      0.93        33
        204       0.67      0.89      0.76        38
        205       0.45      0.51      0.48        75
        206       0.96      0.92      0.94       101
        207       0.26      0.22      0.24        45
        208       0.79      0.77      0.78       108
        209       0.75      0.84      0.79        79
        210       0.65      0.94      0.77        32
        211       0.89      0.90      0.89        60
        212       0.71      0.71      0.71        93
        213       0.75      0.78      0.76        87
        214       0.23      0.31      0.27        35
        215       0.56      0.54      0.55        91
        216       0.71      0.53      0.61        75
        217       0.53      0.63      0.58        70
        218       0.54      0.62      0.58        55
        219       0.94      0.73      0.82        60
        220       0.75      0.92      0.83        50
        221       0.58      0.69      0.63        42
        222       1.00      0.96      0.98        45
        223       0.82      0.71      0.76        96
        224       0.92      0.85      0.88        85
        225       0.51      0.84      0.64        32
        226       0.98      0.93      0.95        43
        227       0.93      0.98      0.95        54
        228       0.44      0.83      0.57        30
        229       0.27      0.53      0.36        32
        230       0.53      0.57      0.55        37
        231       0.83      0.90      0.86        59
        232       0.96      0.84      0.90        97
        233       0.97      0.93      0.95        69
        234       0.90      0.55      0.68        69
        235       0.86      0.75      0.80       103
        236       0.99      0.98      0.98        98
        237       0.93      0.71      0.80       123
        238       0.86      0.89      0.87        93
        239       1.00      0.96      0.98        48
        240       0.60      0.65      0.63        81
        241       0.83      0.94      0.88        32
        242       0.86      0.74      0.80        95
        243       0.74      0.86      0.79        63
        244       0.39      0.55      0.46        40
        245       0.77      0.77      0.77       103
        246       0.71      0.47      0.56        77
        247       0.94      1.00      0.97        61
        248       0.96      0.90      0.93        78
        249       0.93      0.83      0.88        60
        250       0.94      0.92      0.93       120
        251       0.53      0.77      0.63        30
        252       0.82      0.86      0.84        42
        253       0.59      0.66      0.62        44
        254       0.23      0.30      0.26        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.81      0.94      0.87        31
        258       0.36      0.31      0.33        55
        259       0.95      0.91      0.93       128
        260       0.58      0.41      0.48       121
        261       0.62      0.85      0.72        39
        262       0.98      0.98      0.98        62
        263       0.57      0.70      0.63        63
        264       0.89      0.89      0.89        45
        265       0.70      0.91      0.79        55
        266       0.90      0.94      0.92        67
        267       0.84      0.79      0.81        66
        268       0.81      0.70      0.75       148
        269       0.47      0.90      0.62        30
        270       0.55      0.44      0.49       149
        271       0.80      0.93      0.86        42
        272       0.98      0.95      0.97       119
        273       0.88      1.00      0.93        43
        274       0.79      0.73      0.76       167
        275       0.32      0.43      0.37        37
        276       0.22      0.40      0.28        35
        277       0.90      0.96      0.93        57
        278       0.43      0.28      0.34        32
        279       0.47      0.52      0.49        62
        280       0.00      0.00      0.00        66
        281       0.66      0.87      0.75        45
        282       0.84      0.90      0.87        48
        283       0.75      0.97      0.85        31
        284       0.70      0.59      0.64        96
        285       0.75      0.75      0.75       117
        286       0.95      0.85      0.90        66
        287       0.75      0.71      0.73       107
        288       0.72      0.89      0.79        35
        289       0.96      0.92      0.94        87
        290       0.71      0.73      0.72        66
        291       0.93      0.89      0.91        96
        292       0.97      0.70      0.82       135
        293       0.58      0.88      0.70        43
        294       0.46      0.90      0.61        31
        295       0.77      0.84      0.80        56
        296       0.45      0.52      0.48        81
        297       0.90      0.88      0.89       107
        298       0.66      0.71      0.68        41
        299       0.66      0.87      0.75        31
        300       0.80      0.80      0.80        41
        301       0.70      0.81      0.75        52
        302       0.76      0.91      0.83        90
        303       0.46      0.82      0.59        34
        304       0.93      0.95      0.94        41
        305       0.96      0.84      0.89       101
        306       0.95      0.95      0.95        38
        307       0.88      0.67      0.76       120
        308       0.97      0.96      0.96       117
        309       0.93      0.86      0.89        97
        310       0.95      0.96      0.95       109
        311       0.38      0.46      0.42        35
        312       0.95      0.92      0.93        98
        313       0.67      0.90      0.76       116
        314       0.49      0.72      0.58        39
        315       0.69      0.80      0.74        30
        316       0.93      0.91      0.92       155
        317       0.73      0.88      0.80        75
        318       0.70      0.65      0.68        72
        319       0.98      0.86      0.91       154
        320       0.88      0.82      0.85        85
        321       0.07      0.03      0.05        58
        322       0.76      0.79      0.77       119
        323       1.00      1.00      1.00        43
        324       0.91      0.96      0.94        55
        325       0.58      0.55      0.57       112
        326       0.89      0.88      0.88       114
        327       0.80      0.80      0.80        92
        328       0.67      0.69      0.68        84
        329       0.28      0.12      0.17        75
        330       0.64      0.67      0.65        66
        331       0.96      0.86      0.90       127
        332       0.78      0.88      0.83        33
        333       0.48      0.55      0.51        55
        334       0.50      0.29      0.37        41
        335       0.81      0.77      0.79       103
        336       0.73      0.89      0.80        45
        337       0.89      0.93      0.91        54
        338       0.78      0.88      0.83        51
        339       0.72      0.72      0.72        57
        340       0.74      0.76      0.75       127
        341       0.93      0.95      0.94        41
        342       0.74      0.67      0.70       118
        343       0.62      0.67      0.65        45
        344       0.62      0.89      0.73        37
        345       0.99      0.97      0.98       152
        346       0.63      0.70      0.67        64
        347       0.87      0.92      0.89        63
        348       0.79      0.58      0.67        98
        349       1.00      0.98      0.99       104
        350       0.74      0.84      0.79        86
        351       0.88      0.86      0.87       124
        352       0.42      0.59      0.49        63
        353       0.52      0.52      0.52        50
        354       0.47      0.68      0.56        40
        355       0.95      0.87      0.91       108
        356       0.81      0.89      0.85        47
        357       0.71      0.57      0.63        98
        358       0.65      0.74      0.69        42
        359       0.72      0.82      0.77       120
        360       1.00      0.87      0.93       102
        361       0.47      0.45      0.46       120
        362       0.94      0.93      0.94       157
        363       0.93      0.98      0.95        51
        364       0.45      0.67      0.54        42
        365       0.93      0.89      0.91       122
        366       0.72      0.60      0.65       109
        367       1.00      0.97      0.98        67
        368       0.81      0.83      0.82       113
        369       0.56      0.71      0.62        62
        370       0.97      0.92      0.94       130
        371       0.57      0.66      0.61        61
        372       0.76      0.83      0.80       109
        373       0.81      0.76      0.78       145
        374       0.45      0.71      0.55        31
        375       0.51      0.34      0.41        99
        376       0.97      0.92      0.95        38
        377       0.71      0.78      0.74        37
        378       0.22      0.58      0.32        31
        379       0.73      0.81      0.77        54
        380       0.75      0.91      0.82        44
        381       0.23      0.40      0.29        48
        382       0.74      0.90      0.81        31
        383       0.66      0.70      0.68        89
        384       0.99      0.99      0.99       100
        385       0.98      0.89      0.93        45
        386       0.82      0.83      0.83        60
        387       0.62      0.74      0.68        31
        388       0.68      0.69      0.68       115
        389       0.76      0.88      0.82        43
        390       0.99      0.99      0.99       158
        391       0.62      0.56      0.59        89
        392       0.82      0.92      0.87        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.71      0.74      0.72        80
        396       0.64      0.74      0.69        58
        397       0.96      0.92      0.94       121
        398       0.93      0.93      0.93        80
        399       0.95      0.93      0.94        44
        400       0.43      0.68      0.53        37
        401       0.85      0.75      0.80       108
        402       0.93      0.85      0.89        33
        403       0.93      0.88      0.90       123
        404       0.85      0.95      0.90        55
        405       0.90      0.69      0.78       102
        406       0.81      0.84      0.82       110
        407       0.63      0.62      0.62        76
        408       0.64      0.97      0.77        30
        409       0.50      0.60      0.54        42
        410       0.76      0.40      0.53        62
        411       0.70      0.69      0.70       135
        412       0.43      0.52      0.47        64
        413       0.97      1.00      0.98        31
        414       0.95      0.92      0.94        66
        415       0.27      0.25      0.26        88
        416       0.87      0.78      0.83       115
        417       0.96      0.95      0.96       124
        418       0.88      0.77      0.82       126
        419       0.80      0.48      0.60        75
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.49      0.81      0.61        42
        423       0.63      0.67      0.65       109
        424       0.88      1.00      0.94        30
        425       0.80      0.89      0.85        37
        426       0.43      0.65      0.52        31
        427       0.67      0.97      0.79        32
        428       0.50      0.69      0.58        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       1.00      0.98      0.99        42
        432       0.64      0.67      0.66        70
        433       0.38      0.47      0.42        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.54      0.77      0.64        64
        437       0.72      0.88      0.79        32
        438       0.84      0.70      0.76        53
        439       0.70      0.70      0.70        30
        440       0.96      0.95      0.96        85
        441       0.34      0.64      0.44        33
        442       0.80      0.78      0.79       131
        443       0.53      0.58      0.55        85
        444       0.58      0.70      0.63        47
        445       0.85      0.87      0.86        38
        446       0.96      0.84      0.90       122
        447       0.57      0.46      0.51        63
        448       0.85      0.93      0.89        57
        449       0.89      0.98      0.93        57
        450       0.54      0.66      0.60        56
        451       0.65      0.69      0.67        51
        452       0.48      0.50      0.49        40
        453       0.95      0.90      0.92       137
        454       0.90      0.92      0.91        39
        455       0.74      0.87      0.80        63
        456       0.95      0.93      0.94        84
        457       0.92      0.79      0.85       136
        458       0.55      0.94      0.69        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.59        46
        462       0.50      0.44      0.47        68
        463       0.80      0.79      0.79       120
        464       0.86      0.82      0.84        93
        465       0.62      0.84      0.71        62
        466       0.46      0.81      0.59        31
        467       0.73      0.68      0.71        72
        468       0.89      0.98      0.93        41
        469       0.93      0.90      0.92        78
        470       0.97      0.92      0.94        36
        471       0.22      0.05      0.08        44
        472       0.63      0.72      0.67       112
        473       0.89      0.94      0.92        35
        474       0.53      0.54      0.54        78
        475       0.55      0.87      0.68        30
        476       0.91      0.86      0.88        78
        477       0.87      0.78      0.83       106
        478       0.71      0.47      0.57        32
        479       0.77      0.89      0.82        63
        480       0.93      0.98      0.95        91
        481       0.86      0.84      0.85        45
        482       0.86      0.94      0.90        32
        483       0.99      0.90      0.94       166
        484       0.92      1.00      0.96        34
        485       0.60      0.91      0.73        35
        486       0.32      0.33      0.32        73
        487       0.64      0.66      0.65        59
        488       0.43      0.81      0.56        31
        489       0.78      0.94      0.85        48
        490       0.63      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.79      0.81       114
        493       0.98      0.86      0.92       149
        494       0.69      0.65      0.67        51
        495       0.90      0.88      0.89        80
        496       0.93      1.00      0.96        38
        497       0.47      0.49      0.48        88

avg / total       0.77      0.76      0.76     35492

Iteration number, batch number :  3 35
Iteration number, batch number :  3 34
Iteration number, batch number :  3 33
Iteration number, batch number :  3 32
Iteration number, batch number :  3 31
Iteration number, batch number :  3 30
Iteration number, batch number :  3 29
Iteration number, batch number :  3 28
Iteration number, batch number :  3 27
Iteration number, batch number :  3 26
Iteration number, batch number :  3 25
Iteration number, batch number :  3 24
Iteration number, batch number :  3 23
Iteration number, batch number :  3 22
Iteration number, batch number :  3 21
Iteration number, batch number :  3 20
Iteration number, batch number :  3 19
Iteration number, batch number :  3 18
Iteration number, batch number :  3 17
Iteration number, batch number :  3 16
Iteration number, batch number :  3 15
Iteration number, batch number :  3 14
Iteration number, batch number :  3 13
Iteration number, batch number :  3 12
Iteration number, batch number :  3 11
Iteration number, batch number :  3 10
Iteration number, batch number :  3 9
Iteration number, batch number :  3 8
Iteration number, batch number :  3 7
Iteration number, batch number :  3 6
Iteration number, batch number :  3 5
Iteration number, batch number :  3 4
Iteration number, batch number :  3 3
Iteration number, batch number :  3 2
Iteration number, batch number :  3 1
Iteration number, batch number :  3 0
Accuracy on cv data :  78.8843506966 28198.0 28198
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.73      0.90      0.81        42
          2       0.64      0.92      0.76        37
          3       0.96      1.00      0.98        53
          4       0.97      0.83      0.90       131
          5       0.87      0.92      0.89        37
          6       0.43      0.65      0.52        51
          7       0.67      0.55      0.60        47
          8       0.65      0.53      0.59        77
          9       0.98      1.00      0.99        51
         10       0.95      0.91      0.93        46
         11       0.76      0.88      0.81        32
         12       0.86      0.94      0.90        79
         13       0.76      0.89      0.82        85
         14       0.92      0.97      0.95       143
         15       0.46      0.34      0.39       121
         16       0.85      0.94      0.89        35
         17       0.84      0.74      0.79       125
         18       0.89      0.79      0.83        70
         19       0.65      0.76      0.70        92
         20       0.66      0.74      0.70        39
         21       0.96      0.94      0.95        53
         22       0.67      0.70      0.68        46
         23       0.59      0.91      0.71        33
         24       0.73      0.90      0.81        40
         25       0.98      0.98      0.98        86
         26       0.96      0.90      0.93       132
         27       0.83      0.82      0.83        90
         28       0.52      0.84      0.64        31
         29       0.88      0.85      0.87        96
         30       0.86      0.88      0.87        74
         31       0.97      0.96      0.97        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.58      0.78      0.67        32
         35       0.88      0.90      0.89        77
         36       0.91      0.97      0.94        66
         37       0.75      0.73      0.74       115
         38       0.17      0.04      0.07       121
         39       0.94      0.91      0.93        35
         40       0.34      0.41      0.37        76
         41       0.89      0.83      0.86        93
         42       0.66      0.83      0.74        59
         43       0.90      0.95      0.92        63
         44       0.70      0.84      0.76        31
         45       0.83      0.95      0.88        40
         46       0.86      0.61      0.71       161
         47       0.58      0.73      0.65        48
         48       0.91      0.84      0.87        37
         49       0.74      0.89      0.81        38
         50       0.83      0.73      0.78       124
         51       0.53      0.41      0.46       101
         52       0.96      0.96      0.96        51
         53       0.96      0.99      0.97        77
         54       0.94      0.94      0.94        71
         55       1.00      0.95      0.97        95
         56       0.59      0.53      0.56        66
         57       0.81      0.75      0.78        80
         58       0.97      0.99      0.98        99
         59       0.98      0.90      0.94        93
         60       0.94      0.94      0.94        34
         61       0.85      1.00      0.92        41
         62       0.72      0.84      0.78        31
         63       0.99      0.91      0.94        74
         64       0.82      0.97      0.89        38
         65       0.91      0.88      0.89        96
         66       0.79      0.87      0.83        95
         67       0.86      0.90      0.88       132
         68       0.56      0.55      0.55        73
         69       0.40      0.40      0.40        30
         70       0.76      0.88      0.82        43
         71       0.61      0.57      0.59        44
         72       0.46      0.62      0.53        42
         73       0.97      0.84      0.90        37
         74       0.70      0.86      0.77        99
         75       0.88      0.87      0.88       164
         76       0.93      0.95      0.94        65
         77       0.96      0.96      0.96        71
         78       0.93      0.76      0.84        34
         79       0.74      0.85      0.79        33
         80       0.75      0.83      0.79        72
         81       0.92      0.98      0.95       132
         82       0.83      0.66      0.73       175
         83       0.73      0.75      0.74        48
         84       0.42      0.44      0.43        45
         85       0.98      0.96      0.97       110
         86       0.63      0.54      0.58        98
         87       0.78      0.75      0.76        83
         88       0.86      0.88      0.87        43
         89       0.82      0.89      0.85        35
         90       0.79      0.82      0.81        38
         91       0.68      0.53      0.59        51
         92       0.32      0.46      0.38        41
         93       0.88      0.92      0.90        48
         94       0.94      0.87      0.91        71
         95       0.81      0.81      0.81       117
         96       0.64      0.89      0.75        38
         97       0.65      0.84      0.73        31
         98       0.91      0.86      0.89        99
         99       0.78      0.75      0.76        69
        100       0.90      0.91      0.90        99
        101       0.65      0.61      0.63        85
        102       0.49      0.61      0.54        56
        103       0.94      0.94      0.94        36
        104       0.67      0.76      0.71        41
        105       0.70      0.65      0.67        65
        106       0.68      0.81      0.74        32
        107       0.93      0.74      0.83        35
        108       0.67      0.72      0.69        82
        109       0.94      0.79      0.86        78
        110       0.98      0.96      0.97       121
        111       0.63      0.76      0.69        54
        112       0.97      0.96      0.96        89
        113       0.72      0.94      0.82        31
        114       0.85      0.87      0.86       127
        115       0.96      0.94      0.95        78
        116       0.57      0.48      0.52       108
        117       0.84      0.87      0.86       110
        118       0.54      0.42      0.47        88
        119       0.89      0.78      0.83       107
        120       0.54      0.60      0.57        70
        121       0.77      0.95      0.85        39
        122       0.95      0.84      0.89       124
        123       0.82      0.76      0.79        59
        124       0.94      0.96      0.95        70
        125       0.77      0.92      0.84        39
        126       0.78      0.72      0.75       114
        127       0.98      1.00      0.99        56
        128       0.69      0.84      0.76        83
        129       0.97      0.91      0.94       125
        130       0.77      0.83      0.80        48
        131       0.85      0.88      0.86       112
        132       1.00      0.98      0.99        97
        133       0.99      0.74      0.85       164
        134       0.42      0.56      0.48        32
        135       0.83      0.88      0.85        81
        136       0.75      0.74      0.74       104
        137       0.72      0.69      0.70       134
        138       0.99      0.93      0.96        88
        139       0.24      0.13      0.17        38
        140       1.00      0.91      0.95        44
        141       0.90      0.84      0.87        62
        142       1.00      0.92      0.96       104
        143       0.61      0.69      0.65        51
        144       0.52      0.60      0.56        73
        145       0.69      0.88      0.77        50
        146       0.85      0.83      0.84        86
        147       0.87      0.88      0.88        93
        148       0.90      0.78      0.83        58
        149       0.91      0.90      0.90       110
        150       0.79      0.79      0.79        77
        151       0.64      0.63      0.63        75
        152       0.88      0.96      0.92        54
        153       0.48      0.97      0.64        32
        154       0.77      0.78      0.78        74
        155       0.87      0.71      0.78       189
        156       0.63      0.46      0.53        92
        157       0.91      0.94      0.92        31
        158       0.80      0.70      0.75       101
        159       0.53      0.57      0.55       109
        160       0.19      0.26      0.22        39
        161       0.64      0.74      0.68        57
        162       0.89      0.97      0.93        33
        163       0.38      0.34      0.36        70
        164       0.97      0.86      0.91       160
        165       0.37      0.38      0.37        40
        166       0.88      0.93      0.91        72
        167       0.76      0.84      0.80        64
        168       0.88      0.75      0.81        76
        169       0.76      0.85      0.80        40
        170       0.99      0.95      0.97       141
        171       0.15      0.06      0.09        62
        172       0.86      0.88      0.87        89
        173       0.64      0.72      0.67        39
        174       0.76      0.92      0.83        38
        175       0.74      0.96      0.83        45
        176       0.40      0.58      0.47        43
        177       0.66      0.84      0.74        32
        178       0.94      0.89      0.92        55
        179       0.40      0.53      0.46        32
        180       0.66      0.63      0.65        82
        181       0.62      0.62      0.62        32
        182       0.73      0.68      0.70        87
        183       0.65      0.86      0.74        42
        184       0.87      0.90      0.88        58
        185       0.72      0.87      0.79        30
        186       0.95      0.95      0.95        56
        187       0.68      0.97      0.80        31
        188       0.97      1.00      0.99        33
        189       0.78      0.91      0.84        47
        190       0.97      0.93      0.95       108
        191       0.98      1.00      0.99        40
        192       0.94      0.94      0.94       119
        193       0.94      1.00      0.97        33
        194       0.77      0.81      0.79       152
        195       0.95      0.83      0.89        71
        196       0.92      0.92      0.92        39
        197       0.94      0.94      0.94       124
        198       0.89      0.77      0.83        31
        199       0.42      0.55      0.48        38
        200       0.94      0.97      0.96        34
        201       0.33      0.35      0.34        31
        202       0.83      0.85      0.84        53
        203       0.88      0.88      0.88        34
        204       0.58      0.77      0.66        39
        205       0.62      0.69      0.65        75
        206       0.98      0.88      0.93       101
        207       0.35      0.33      0.34        45
        208       0.81      0.71      0.76       108
        209       0.89      0.94      0.91        80
        210       0.74      0.85      0.79        33
        211       0.86      0.95      0.90        60
        212       0.73      0.66      0.69        93
        213       0.79      0.77      0.78        87
        214       0.22      0.17      0.19        36
        215       0.47      0.47      0.47        92
        216       0.76      0.83      0.79        75
        217       0.53      0.58      0.55        71
        218       0.49      0.61      0.54        56
        219       0.87      0.75      0.80        60
        220       0.81      0.84      0.82        50
        221       0.60      0.65      0.62        43
        222       1.00      1.00      1.00        46
        223       0.72      0.71      0.72        97
        224       1.00      0.84      0.91        85
        225       0.43      0.73      0.54        33
        226       0.91      1.00      0.96        43
        227       0.96      0.94      0.95        54
        228       0.53      0.80      0.64        30
        229       0.37      0.69      0.48        32
        230       0.55      0.57      0.56        37
        231       0.92      0.81      0.86        59
        232       0.89      0.86      0.88        98
        233       0.94      0.94      0.94        70
        234       0.96      0.97      0.96        69
        235       0.87      0.86      0.86       104
        236       1.00      0.96      0.98        99
        237       0.92      0.85      0.88       123
        238       0.88      0.89      0.89        93
        239       1.00      0.96      0.98        48
        240       0.57      0.61      0.59        82
        241       0.76      0.85      0.80        33
        242       0.83      0.85      0.84        95
        243       0.81      0.87      0.84        63
        244       0.42      0.54      0.47        41
        245       0.78      0.69      0.73       104
        246       0.87      0.70      0.78        77
        247       0.95      1.00      0.98        62
        248       0.96      0.95      0.95        78
        249       0.90      0.74      0.81        61
        250       0.92      0.83      0.87       121
        251       0.61      0.55      0.58        31
        252       0.72      0.90      0.80        42
        253       0.56      0.62      0.59        45
        254       0.20      0.27      0.23        48
        255       1.00      1.00      1.00        42
        256       1.00      0.98      0.99        98
        257       0.89      0.97      0.93        32
        258       0.46      0.53      0.49        55
        259       0.92      0.95      0.93       128
        260       0.62      0.48      0.54       121
        261       0.56      0.82      0.67        39
        262       0.98      1.00      0.99        62
        263       0.51      0.64      0.57        64
        264       0.93      0.93      0.93        45
        265       0.87      0.96      0.92        56
        266       0.89      0.94      0.91        68
        267       0.84      0.88      0.86        67
        268       0.85      0.81      0.83       148
        269       0.58      0.81      0.68        31
        270       0.66      0.42      0.51       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.96      1.00      0.98        44
        274       0.79      0.76      0.77       168
        275       0.32      0.63      0.42        38
        276       0.24      0.49      0.32        35
        277       0.86      0.93      0.89        58
        278       0.69      0.76      0.72        33
        279       0.50      0.57      0.53        63
        280       0.97      0.88      0.92        67
        281       0.67      0.82      0.74        45
        282       0.77      0.94      0.84        49
        283       0.76      1.00      0.86        32
        284       0.73      0.58      0.65        96
        285       0.76      0.81      0.78       118
        286       0.84      0.86      0.85        66
        287       0.72      0.67      0.70       107
        288       0.89      0.94      0.92        35
        289       0.94      0.86      0.90        88
        290       0.81      0.83      0.82        66
        291       0.99      0.88      0.93        96
        292       0.98      0.81      0.89       135
        293       0.74      0.80      0.77        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.63      0.59      0.61        82
        297       0.90      0.82      0.86       107
        298       0.70      0.78      0.74        41
        299       0.86      0.81      0.83        31
        300       0.80      0.88      0.84        42
        301       0.75      0.83      0.79        53
        302       0.95      0.93      0.94        90
        303       0.55      0.77      0.64        35
        304       0.91      0.98      0.94        42
        305       0.95      0.88      0.91       102
        306       1.00      0.95      0.97        39
        307       0.88      0.76      0.81       120
        308       0.95      0.99      0.97       117
        309       0.93      0.84      0.88        98
        310       0.96      0.99      0.98       110
        311       0.46      0.44      0.45        36
        312       0.96      1.00      0.98        99
        313       0.79      0.94      0.86       116
        314       0.59      0.69      0.64        39
        315       0.62      0.70      0.66        30
        316       0.93      0.83      0.88       156
        317       0.73      0.87      0.79        75
        318       0.61      0.71      0.66        72
        319       0.97      0.90      0.93       154
        320       0.95      0.85      0.90        86
        321       0.26      0.17      0.20        59
        322       0.75      0.79      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.88      0.91        56
        325       0.56      0.54      0.55       112
        326       0.90      0.90      0.90       114
        327       0.86      0.85      0.85        93
        328       0.75      0.69      0.72        85
        329       0.27      0.12      0.17        76
        330       0.58      0.67      0.62        66
        331       0.89      0.89      0.89       128
        332       0.88      0.91      0.90        33
        333       0.48      0.55      0.51        56
        334       0.62      0.44      0.51        41
        335       0.87      0.64      0.74       103
        336       0.78      0.83      0.80        46
        337       0.96      0.93      0.94        54
        338       0.76      0.92      0.83        52
        339       0.71      0.68      0.70        57
        340       0.81      0.75      0.78       128
        341       0.89      0.93      0.91        42
        342       0.80      0.62      0.70       118
        343       0.56      0.56      0.56        45
        344       0.57      0.71      0.64        38
        345       0.99      0.97      0.98       153
        346       0.59      0.78      0.67        65
        347       0.91      0.83      0.87        64
        348       0.77      0.56      0.65        99
        349       1.00      0.99      1.00       105
        350       0.75      0.80      0.78        86
        351       0.93      0.95      0.94       125
        352       0.43      0.36      0.39        64
        353       0.39      0.54      0.45        50
        354       0.39      0.66      0.49        41
        355       0.92      0.85      0.88       108
        356       0.80      0.75      0.77        48
        357       0.67      0.65      0.66        98
        358       0.68      0.65      0.67        43
        359       0.71      0.72      0.72       120
        360       1.00      0.98      0.99       103
        361       0.40      0.42      0.41       120
        362       0.98      0.89      0.93       158
        363       0.89      0.98      0.93        51
        364       0.53      0.74      0.62        42
        365       0.89      0.86      0.88       122
        366       0.72      0.57      0.64       110
        367       1.00      0.94      0.97        67
        368       0.85      0.87      0.86       114
        369       0.67      0.82      0.74        62
        370       0.92      0.92      0.92       130
        371       0.67      0.66      0.67        62
        372       0.78      0.84      0.81       110
        373       0.89      0.86      0.87       145
        374       0.47      0.75      0.58        32
        375       0.48      0.37      0.42       100
        376       0.97      0.95      0.96        39
        377       0.64      0.74      0.68        38
        378       0.34      0.52      0.41        31
        379       0.74      0.83      0.78        54
        380       0.73      0.86      0.79        44
        381       0.30      0.35      0.32        48
        382       0.68      0.84      0.75        31
        383       0.70      0.78      0.74        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.75      0.79      0.77        61
        387       0.61      0.72      0.66        32
        388       0.70      0.66      0.68       115
        389       0.74      0.86      0.80        43
        390       0.98      0.97      0.98       158
        391       0.62      0.72      0.67        89
        392       0.84      0.96      0.90        50
        393       0.97      0.93      0.95        69
        394       0.84      0.87      0.86        55
        395       0.80      0.84      0.82        81
        396       0.71      0.75      0.73        59
        397       0.94      0.97      0.95       121
        398       0.89      0.94      0.91        80
        399       0.83      0.87      0.85        45
        400       0.53      0.50      0.51        38
        401       0.76      0.84      0.80       108
        402       0.91      0.91      0.91        34
        403       0.87      0.84      0.85       123
        404       0.82      0.89      0.85        55
        405       0.88      0.82      0.85       103
        406       0.77      0.87      0.82       111
        407       0.73      0.74      0.73        76
        408       0.59      0.87      0.70        30
        409       0.57      0.62      0.59        42
        410       0.73      0.48      0.58        63
        411       0.79      0.74      0.76       136
        412       0.58      0.52      0.55        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.38      0.38        88
        416       0.90      0.71      0.79       116
        417       0.94      0.94      0.94       124
        418       0.89      0.72      0.80       127
        419       0.82      0.89      0.86        76
        420       0.88      0.92      0.90        48
        421       0.91      1.00      0.95        30
        422       0.63      0.81      0.71        42
        423       0.75      0.72      0.73       110
        424       0.69      0.97      0.81        30
        425       0.77      0.89      0.82        37
        426       0.69      0.78      0.74        32
        427       0.75      0.94      0.83        32
        428       0.46      0.65      0.54        55
        429       0.96      0.97      0.96        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.73      0.66      0.70        71
        433       0.49      0.49      0.49        55
        434       0.88      0.94      0.91        32
        435       0.92      0.82      0.86        66
        436       0.92      0.86      0.89        65
        437       0.70      0.94      0.80        32
        438       0.74      0.65      0.69        54
        439       0.78      0.70      0.74        30
        440       0.99      0.97      0.98        86
        441       0.39      0.64      0.48        33
        442       0.81      0.77      0.79       131
        443       0.52      0.63      0.57        86
        444       0.52      0.46      0.49        48
        445       0.76      0.92      0.83        38
        446       0.97      0.96      0.96       123
        447       0.61      0.68      0.64        63
        448       0.85      0.78      0.81        58
        449       0.89      0.95      0.92        57
        450       0.59      0.73      0.66        56
        451       0.69      0.73      0.70        51
        452       0.53      0.71      0.60        41
        453       0.96      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.76      0.87      0.81        63
        456       0.94      0.78      0.85        85
        457       0.92      0.78      0.85       137
        458       0.73      0.77      0.75        47
        459       0.62      0.38      0.47        87
        460       0.99      0.99      0.99        95
        461       0.52      0.74      0.61        46
        462       0.56      0.45      0.50        69
        463       0.89      0.71      0.79       120
        464       0.87      0.82      0.84        93
        465       0.71      0.89      0.79        63
        466       0.35      0.50      0.41        32
        467       0.76      0.68      0.72        73
        468       0.93      0.98      0.95        42
        469       0.90      0.90      0.90        78
        470       1.00      0.97      0.99        36
        471       0.76      0.66      0.71        44
        472       0.69      0.63      0.66       112
        473       0.87      0.94      0.91        36
        474       0.55      0.46      0.50        79
        475       0.46      0.74      0.57        31
        476       0.97      0.88      0.93        78
        477       0.93      0.82      0.87       107
        478       0.95      0.58      0.72        33
        479       0.88      0.94      0.91        63
        480       0.91      0.95      0.93        91
        481       0.74      0.76      0.75        46
        482       0.97      0.88      0.92        33
        483       0.99      0.84      0.91       167
        484       0.97      0.97      0.97        35
        485       0.82      0.75      0.78        36
        486       0.28      0.30      0.29        74
        487       0.72      0.78      0.75        60
        488       0.55      0.68      0.61        31
        489       0.92      0.92      0.92        49
        490       0.80      0.63      0.70       121
        491       0.82      0.85      0.84        39
        492       0.73      0.72      0.73       114
        493       0.95      0.85      0.89       149
        494       0.65      0.71      0.68        51
        495       0.91      0.93      0.92        80
        496       0.79      0.97      0.87        39
        497       0.57      0.44      0.50        88

avg / total       0.80      0.79      0.79     35746

Iteration number, batch number :  4 0
Training data accuracy :  0.834337349398
Training data loss     :  0.0054129070063
Iteration number, batch number :  4 1
Training data accuracy :  0.821285140562
Training data loss     :  0.00549616991623
Iteration number, batch number :  4 2
Training data accuracy :  0.827309236948
Training data loss     :  0.00544854129343
Iteration number, batch number :  4 3
Training data accuracy :  0.832329317269
Training data loss     :  0.00545686940116
Iteration number, batch number :  4 4
Training data accuracy :  0.839357429719
Training data loss     :  0.00541888998395
Iteration number, batch number :  4 5
Training data accuracy :  0.852409638554
Training data loss     :  0.00523669668975
Iteration number, batch number :  4 6
Training data accuracy :  0.832329317269
Training data loss     :  0.00563556096766
Iteration number, batch number :  4 7
Training data accuracy :  0.850401606426
Training data loss     :  0.00482843243083
Iteration number, batch number :  4 8
Training data accuracy :  0.827309236948
Training data loss     :  0.00527986086309
Iteration number, batch number :  4 9
Training data accuracy :  0.840361445783
Training data loss     :  0.00533524981749
Iteration number, batch number :  4 10
Training data accuracy :  0.859437751004
Training data loss     :  0.00478468899399
Iteration number, batch number :  4 11
Training data accuracy :  0.838353413655
Training data loss     :  0.00518305597108
Iteration number, batch number :  4 12
Training data accuracy :  0.85140562249
Training data loss     :  0.0051380285025
Iteration number, batch number :  4 13
Training data accuracy :  0.830321285141
Training data loss     :  0.00558021564123
Iteration number, batch number :  4 14
Training data accuracy :  0.842369477912
Training data loss     :  0.00493358040815
Iteration number, batch number :  4 15
Training data accuracy :  0.850401606426
Training data loss     :  0.00484899314283
Iteration number, batch number :  4 16
Training data accuracy :  0.867469879518
Training data loss     :  0.00454028233932
Iteration number, batch number :  4 17
Training data accuracy :  0.840361445783
Training data loss     :  0.0048350670754
Iteration number, batch number :  4 18
Training data accuracy :  0.84437751004
Training data loss     :  0.00523579439145
Iteration number, batch number :  4 19
Training data accuracy :  0.85843373494
Training data loss     :  0.00504623114793
Iteration number, batch number :  4 20
Training data accuracy :  0.864457831325
Training data loss     :  0.00474355075744
Iteration number, batch number :  4 21
Training data accuracy :  0.833333333333
Training data loss     :  0.00486031924396
Iteration number, batch number :  4 22
Training data accuracy :  0.853413654618
Training data loss     :  0.00460441150137
Iteration number, batch number :  4 23
Training data accuracy :  0.845381526104
Training data loss     :  0.00459528954375
Iteration number, batch number :  4 24
Training data accuracy :  0.845381526104
Training data loss     :  0.005038072327
Iteration number, batch number :  4 25
Training data accuracy :  0.850401606426
Training data loss     :  0.00511174264762
Iteration number, batch number :  4 26
Training data accuracy :  0.860441767068
Training data loss     :  0.00482345116879
Iteration number, batch number :  4 27
Training data accuracy :  0.845381526104
Training data loss     :  0.00490002703063
Iteration number, batch number :  4 28
Training data accuracy :  0.871485943775
Training data loss     :  0.00449048910627
Iteration number, batch number :  4 29
Training data accuracy :  0.84437751004
Training data loss     :  0.00496769647555
Iteration number, batch number :  4 30
Training data accuracy :  0.848393574297
Training data loss     :  0.00480829076815
Iteration number, batch number :  4 31
Training data accuracy :  0.843373493976
Training data loss     :  0.00505785903676
Iteration number, batch number :  4 32
Training data accuracy :  0.853413654618
Training data loss     :  0.00494621419419
Iteration number, batch number :  4 33
Training data accuracy :  0.883534136546
Training data loss     :  0.00453589463265
Iteration number, batch number :  4 34
Training data accuracy :  0.875502008032
Training data loss     :  0.00440019078652
Iteration number, batch number :  4 35
Training data accuracy :  0.868473895582
Training data loss     :  0.00426501127008
Iteration number, batch number :  4 36
Training data accuracy :  0.855421686747
Training data loss     :  0.0044902920094
Iteration number, batch number :  4 37
Training data accuracy :  0.868473895582
Training data loss     :  0.00456044046587
Iteration number, batch number :  4 38
Training data accuracy :  0.868473895582
Training data loss     :  0.00459584372815
Iteration number, batch number :  4 39
Training data accuracy :  0.889558232932
Training data loss     :  0.0040289494263
Iteration number, batch number :  4 40
Training data accuracy :  0.853413654618
Training data loss     :  0.0049349456866
Iteration number, batch number :  4 41
Training data accuracy :  0.85843373494
Training data loss     :  0.0045331264315
Iteration number, batch number :  4 42
Training data accuracy :  0.873493975904
Training data loss     :  0.00449201526758
Iteration number, batch number :  4 43
Training data accuracy :  0.872489959839
Training data loss     :  0.00449704848241
Iteration number, batch number :  4 44
Training data accuracy :  0.868473895582
Training data loss     :  0.00412514401891
Iteration number, batch number :  4 45
Training data accuracy :  0.866465863454
Training data loss     :  0.00433021027893
Iteration number, batch number :  4 46
2017-07-01 15:18:42.759617: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 13678610 get requests, put_count=13678340 evicted_count=9000 eviction_rate=0.000657975 and unsatisfied allocation rate=0.000740134
Training data accuracy :  0.86546184739
Training data loss     :  0.00430739635203
Iteration number, batch number :  4 47
Training data accuracy :  0.874497991968
Training data loss     :  0.00464534881418
Iteration number, batch number :  4 48
Training data accuracy :  0.856425702811
Training data loss     :  0.00481854548715
Iteration number, batch number :  4 49
Training data accuracy :  0.855421686747
Training data loss     :  0.00450410881096
Iteration number, batch number :  4 50
Training data accuracy :  0.869477911647
Training data loss     :  0.00454381169331
Iteration number, batch number :  4 51
Training data accuracy :  0.839357429719
Training data loss     :  0.00494335743133
Iteration number, batch number :  4 52
Training data accuracy :  0.846385542169
Training data loss     :  0.00542794703672
Iteration number, batch number :  4 53
Training data accuracy :  0.84437751004
Training data loss     :  0.00515366394343
Iteration number, batch number :  4 54
Training data accuracy :  0.848393574297
Training data loss     :  0.00510764652758
Iteration number, batch number :  4 55
Training data accuracy :  0.838353413655
Training data loss     :  0.00494130404788
Iteration number, batch number :  4 56
Training data accuracy :  0.85140562249
Training data loss     :  0.0051731151178
Iteration number, batch number :  4 57
Training data accuracy :  0.834337349398
Training data loss     :  0.00528324431662
Iteration number, batch number :  4 58
Training data accuracy :  0.824297188755
Training data loss     :  0.00527679640886
Iteration number, batch number :  4 59
Training data accuracy :  0.842369477912
Training data loss     :  0.00552477883132
Iteration number, batch number :  4 60
Training data accuracy :  0.841365461847
Training data loss     :  0.00521280520126
Iteration number, batch number :  4 61
Training data accuracy :  0.830321285141
Training data loss     :  0.00537217101218
Iteration number, batch number :  4 62
Training data accuracy :  0.836345381526
Training data loss     :  0.00531967206754
Iteration number, batch number :  4 63
Training data accuracy :  0.828313253012
Training data loss     :  0.00545526579079
Iteration number, batch number :  4 64
Training data accuracy :  0.845381526104
Training data loss     :  0.00569336816597
Iteration number, batch number :  4 65
Training data accuracy :  0.833333333333
Training data loss     :  0.00587175415358
Iteration number, batch number :  4 66
Training data accuracy :  0.83734939759
Training data loss     :  0.00522446490661
Iteration number, batch number :  4 67
Training data accuracy :  0.854417670683
Training data loss     :  0.0053598298219
Iteration number, batch number :  4 68
Training data accuracy :  0.838353413655
Training data loss     :  0.00559930346397
Iteration number, batch number :  4 69
Training data accuracy :  0.832329317269
Training data loss     :  0.00531083221044
Iteration number, batch number :  4 35
Iteration number, batch number :  4 34
Iteration number, batch number :  4 33
Iteration number, batch number :  4 32
Iteration number, batch number :  4 31
Iteration number, batch number :  4 30
Iteration number, batch number :  4 29
Iteration number, batch number :  4 28
Iteration number, batch number :  4 27
Iteration number, batch number :  4 26
Iteration number, batch number :  4 25
Iteration number, batch number :  4 24
Iteration number, batch number :  4 23
Iteration number, batch number :  4 22
Iteration number, batch number :  4 21
Iteration number, batch number :  4 20
Iteration number, batch number :  4 19
Iteration number, batch number :  4 18
Iteration number, batch number :  4 17
Iteration number, batch number :  4 16
Iteration number, batch number :  4 15
Iteration number, batch number :  4 14
Iteration number, batch number :  4 13
Iteration number, batch number :  4 12
Iteration number, batch number :  4 11
Iteration number, batch number :  4 10
Iteration number, batch number :  4 9
Iteration number, batch number :  4 8
Iteration number, batch number :  4 7
Iteration number, batch number :  4 6
Iteration number, batch number :  4 5
Iteration number, batch number :  4 4
Iteration number, batch number :  4 3
Iteration number, batch number :  4 2
Iteration number, batch number :  4 1
Iteration number, batch number :  4 0
Accuracy on test data :  76.1411022202 27024.0 27024
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.77      0.91      0.84        45
          1       0.72      0.95      0.82        41
          2       0.64      0.81      0.71        37
          3       0.83      0.96      0.89        52
          4       0.95      0.54      0.69       130
          5       0.92      0.97      0.95        36
          6       0.47      0.67      0.55        51
          7       0.71      0.62      0.66        47
          8       0.39      0.38      0.39        76
          9       0.98      1.00      0.99        51
         10       0.95      0.93      0.94        45
         11       0.35      0.45      0.39        31
         12       0.94      0.85      0.89        79
         13       0.71      0.84      0.77        85
         14       0.91      0.85      0.88       143
         15       0.51      0.41      0.45       120
         16       0.91      0.94      0.93        34
         17       0.66      0.32      0.43       124
         18       0.85      0.80      0.82        70
         19       0.54      0.56      0.55        91
         20       0.40      0.62      0.48        39
         21       0.91      0.38      0.53        53
         22       0.52      0.51      0.52        45
         23       0.48      0.73      0.58        33
         24       0.79      0.78      0.78        40
         25       0.98      0.98      0.98        85
         26       0.92      0.73      0.81       132
         27       0.93      0.87      0.90        90
         28       0.36      0.53      0.43        30
         29       0.86      0.84      0.85        96
         30       0.86      0.86      0.86        74
         31       0.97      0.89      0.93        79
         32       1.00      0.98      0.99       269
         33       0.97      0.94      0.95        31
         34       0.42      0.66      0.51        32
         35       0.88      0.95      0.91        77
         36       0.87      0.91      0.89        66
         37       0.75      0.83      0.79       114
         38       0.06      0.01      0.01       121
         39       0.82      0.82      0.82        34
         40       0.33      0.45      0.38        76
         41       0.91      0.64      0.75        92
         42       0.65      0.72      0.68        58
         43       0.95      0.87      0.91        63
         44       0.60      0.80      0.69        30
         45       0.83      0.97      0.90        40
         46       0.00      0.00      0.00       161
         47       0.58      0.77      0.66        47
         48       0.89      0.94      0.92        36
         49       0.80      0.95      0.86        37
         50       0.78      0.65      0.71       123
         51       0.60      0.47      0.53       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.95      0.90      0.93        70
         55       0.98      0.94      0.96        94
         56       0.58      0.64      0.61        66
         57       0.68      0.65      0.66        79
         58       0.97      0.92      0.94        98
         59       0.95      0.89      0.92        92
         60       1.00      0.88      0.94        33
         61       0.84      0.93      0.88        41
         62       0.33      0.07      0.11        30
         63       0.83      0.88      0.86        74
         64       0.86      1.00      0.93        38
         65       0.93      0.88      0.91        95
         66       0.77      0.67      0.72        94
         67       0.87      0.90      0.88       132
         68       0.39      0.43      0.41        72
         69       0.38      0.53      0.44        30
         70       0.72      0.81      0.76        42
         71       0.66      0.80      0.72        44
         72       0.34      0.67      0.45        42
         73       0.86      0.84      0.85        37
         74       0.64      0.89      0.74        98
         75       0.95      0.87      0.91       163
         76       0.91      0.91      0.91        65
         77       0.97      0.94      0.96        70
         78       0.85      0.85      0.85        33
         79       0.60      0.79      0.68        33
         80       0.76      0.68      0.72        71
         81       0.96      0.98      0.97       131
         82       0.82      0.69      0.75       174
         83       0.70      0.74      0.72        47
         84       0.24      0.27      0.25        44
         85       0.95      0.93      0.94       110
         86       0.55      0.46      0.50        97
         87       0.92      0.78      0.84        83
         88       0.75      0.84      0.79        43
         89       0.70      0.91      0.79        35
         90       0.72      0.82      0.77        38
         91       0.67      0.61      0.64        51
         92       0.42      0.70      0.53        40
         93       0.96      0.92      0.94        48
         94       0.89      0.90      0.90        71
         95       0.81      0.79      0.80       117
         96       0.55      0.84      0.67        38
         97       0.69      0.90      0.78        30
         98       0.88      0.85      0.86        98
         99       0.62      0.72      0.67        69
        100       0.92      0.91      0.91        98
        101       0.64      0.64      0.64        85
        102       0.37      0.54      0.43        56
        103       0.97      0.92      0.94        36
        104       0.69      0.60      0.64        40
        105       0.65      0.55      0.60        65
        106       0.62      0.72      0.67        32
        107       1.00      0.56      0.72        34
        108       0.69      0.75      0.72        81
        109       0.62      0.26      0.37        77
        110       0.98      0.92      0.95       121
        111       0.55      0.53      0.54        53
        112       0.90      0.89      0.89        88
        113       0.76      0.97      0.85        30
        114       0.83      0.88      0.85       127
        115       0.93      0.73      0.82        78
        116       0.58      0.51      0.54       108
        117       0.93      0.82      0.87       110
        118       0.49      0.40      0.44        88
        119       0.89      0.79      0.84       106
        120       0.61      0.77      0.68        69
        121       0.78      0.82      0.80        39
        122       0.86      0.77      0.82       124
        123       0.83      0.83      0.83        59
        124       1.00      0.96      0.98        69
        125       0.67      0.87      0.76        38
        126       0.74      0.62      0.67       113
        127       0.98      1.00      0.99        55
        128       0.70      0.85      0.77        82
        129       0.93      0.89      0.91       125
        130       0.59      0.85      0.70        47
        131       0.82      0.36      0.50       112
        132       0.98      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.46      0.56      0.51        32
        135       0.80      0.79      0.79        80
        136       0.73      0.71      0.72       103
        137       0.74      0.81      0.77       134
        138       0.95      0.91      0.93        88
        139       0.16      0.08      0.11        37
        140       0.92      1.00      0.96        44
        141       0.85      0.90      0.87        61
        142       0.98      0.94      0.96       103
        143       0.52      0.69      0.59        51
        144       0.47      0.65      0.55        72
        145       0.73      0.98      0.83        49
        146       0.73      0.74      0.74        85
        147       0.90      0.87      0.89        93
        148       0.86      0.86      0.86        57
        149       0.83      0.84      0.84       109
        150       0.89      0.64      0.75        76
        151       0.66      0.69      0.68        75
        152       0.94      0.91      0.92        54
        153       0.50      0.78      0.61        32
        154       0.88      0.88      0.88        73
        155       0.82      0.59      0.69       189
        156       0.62      0.48      0.54        92
        157       0.91      0.94      0.92        31
        158       0.47      0.15      0.23       100
        159       0.51      0.41      0.45       108
        160       0.24      0.44      0.31        39
        161       0.65      0.74      0.69        57
        162       0.94      1.00      0.97        33
        163       0.38      0.47      0.42        70
        164       0.96      0.81      0.87       159
        165       0.50      0.72      0.59        39
        166       0.89      0.94      0.92        71
        167       0.85      0.80      0.82        64
        168       0.85      0.81      0.83        75
        169       0.53      0.64      0.58        39
        170       0.99      0.94      0.96       140
        171       0.15      0.08      0.11        61
        172       0.79      0.90      0.84        88
        173       0.61      0.69      0.65        39
        174       0.82      0.89      0.86        37
        175       0.75      0.80      0.77        45
        176       0.34      0.50      0.41        42
        177       0.61      0.87      0.72        31
        178       0.93      0.95      0.94        55
        179       0.38      0.55      0.45        31
        180       0.61      0.62      0.62        82
        181       0.57      0.65      0.61        31
        182       0.68      0.66      0.67        87
        183       0.72      0.88      0.79        41
        184       0.87      0.91      0.89        57
        185       0.76      0.87      0.81        30
        186       1.00      0.95      0.97        56
        187       0.73      1.00      0.85        30
        188       0.97      1.00      0.99        33
        189       0.70      0.80      0.75        46
        190       0.91      0.99      0.95       107
        191       0.90      0.95      0.93        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.79      0.74      0.76       152
        195       0.92      0.85      0.88        71
        196       0.95      0.95      0.95        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.38      0.73      0.50        37
        200       0.94      1.00      0.97        34
        201       0.16      0.27      0.20        30
        202       0.74      0.81      0.77        53
        203       0.89      0.97      0.93        33
        204       0.65      0.89      0.76        38
        205       0.48      0.52      0.50        75
        206       0.95      0.92      0.93       101
        207       0.28      0.24      0.26        45
        208       0.80      0.77      0.78       108
        209       0.76      0.84      0.80        79
        210       0.70      0.94      0.80        32
        211       0.90      0.90      0.90        60
        212       0.73      0.71      0.72        93
        213       0.76      0.79      0.78        87
        214       0.24      0.34      0.28        35
        215       0.57      0.55      0.56        91
        216       0.74      0.53      0.62        75
        217       0.54      0.61      0.57        70
        218       0.56      0.65      0.61        55
        219       0.92      0.73      0.81        60
        220       0.75      0.92      0.83        50
        221       0.54      0.69      0.60        42
        222       1.00      0.96      0.98        45
        223       0.80      0.71      0.75        96
        224       0.92      0.85      0.88        85
        225       0.52      0.84      0.64        32
        226       0.98      0.93      0.95        43
        227       0.96      0.98      0.97        54
        228       0.42      0.83      0.56        30
        229       0.28      0.53      0.37        32
        230       0.49      0.57      0.53        37
        231       0.84      0.90      0.87        59
        232       0.96      0.82      0.89        97
        233       0.97      0.94      0.96        69
        234       0.90      0.55      0.68        69
        235       0.86      0.75      0.80       103
        236       0.99      0.98      0.98        98
        237       0.92      0.72      0.80       123
        238       0.86      0.89      0.87        93
        239       1.00      0.96      0.98        48
        240       0.62      0.65      0.63        81
        241       0.83      0.94      0.88        32
        242       0.86      0.74      0.80        95
        243       0.75      0.86      0.80        63
        244       0.40      0.55      0.46        40
        245       0.78      0.78      0.78       103
        246       0.71      0.51      0.59        77
        247       0.94      0.98      0.96        61
        248       0.96      0.90      0.93        78
        249       0.91      0.83      0.87        60
        250       0.94      0.93      0.94       120
        251       0.54      0.73      0.62        30
        252       0.82      0.88      0.85        42
        253       0.55      0.66      0.60        44
        254       0.24      0.32      0.28        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.81      0.94      0.87        31
        258       0.38      0.33      0.35        55
        259       0.96      0.91      0.94       128
        260       0.57      0.41      0.48       121
        261       0.63      0.85      0.73        39
        262       0.97      0.98      0.98        62
        263       0.52      0.70      0.59        63
        264       0.87      0.89      0.88        45
        265       0.70      0.91      0.79        55
        266       0.90      0.94      0.92        67
        267       0.86      0.77      0.82        66
        268       0.81      0.69      0.74       148
        269       0.50      0.90      0.64        30
        270       0.57      0.45      0.50       149
        271       0.81      0.93      0.87        42
        272       0.97      0.95      0.96       119
        273       0.88      1.00      0.93        43
        274       0.79      0.74      0.76       167
        275       0.33      0.46      0.39        37
        276       0.19      0.40      0.26        35
        277       0.92      0.96      0.94        57
        278       0.43      0.28      0.34        32
        279       0.48      0.52      0.50        62
        280       0.00      0.00      0.00        66
        281       0.65      0.87      0.74        45
        282       0.86      0.90      0.88        48
        283       0.79      0.97      0.87        31
        284       0.68      0.61      0.64        96
        285       0.76      0.75      0.76       117
        286       0.97      0.85      0.90        66
        287       0.74      0.70      0.72       107
        288       0.69      0.89      0.78        35
        289       0.96      0.92      0.94        87
        290       0.71      0.73      0.72        66
        291       0.94      0.89      0.91        96
        292       0.97      0.70      0.82       135
        293       0.59      0.91      0.72        43
        294       0.46      0.90      0.61        31
        295       0.77      0.84      0.80        56
        296       0.46      0.52      0.49        81
        297       0.89      0.89      0.89       107
        298       0.65      0.68      0.67        41
        299       0.66      0.87      0.75        31
        300       0.80      0.80      0.80        41
        301       0.69      0.81      0.74        52
        302       0.80      0.91      0.85        90
        303       0.45      0.82      0.58        34
        304       0.93      0.95      0.94        41
        305       0.92      0.85      0.89       101
        306       0.92      0.95      0.94        38
        307       0.89      0.67      0.76       120
        308       0.97      0.96      0.96       117
        309       0.91      0.86      0.88        97
        310       0.95      0.95      0.95       109
        311       0.38      0.46      0.42        35
        312       0.94      0.93      0.93        98
        313       0.66      0.91      0.77       116
        314       0.48      0.72      0.58        39
        315       0.75      0.80      0.77        30
        316       0.95      0.91      0.93       155
        317       0.79      0.85      0.82        75
        318       0.70      0.64      0.67        72
        319       0.98      0.86      0.91       154
        320       0.84      0.82      0.83        85
        321       0.08      0.03      0.05        58
        322       0.76      0.80      0.78       119
        323       1.00      1.00      1.00        43
        324       0.93      0.96      0.95        55
        325       0.59      0.57      0.58       112
        326       0.90      0.87      0.88       114
        327       0.82      0.80      0.81        92
        328       0.67      0.69      0.68        84
        329       0.30      0.15      0.20        75
        330       0.65      0.67      0.66        66
        331       0.95      0.86      0.90       127
        332       0.78      0.88      0.83        33
        333       0.47      0.56      0.51        55
        334       0.55      0.29      0.38        41
        335       0.81      0.78      0.79       103
        336       0.73      0.89      0.80        45
        337       0.91      0.91      0.91        54
        338       0.79      0.88      0.83        51
        339       0.70      0.75      0.73        57
        340       0.75      0.76      0.75       127
        341       0.93      0.95      0.94        41
        342       0.73      0.68      0.70       118
        343       0.60      0.67      0.63        45
        344       0.62      0.89      0.73        37
        345       0.99      0.97      0.98       152
        346       0.63      0.69      0.66        64
        347       0.87      0.92      0.89        63
        348       0.76      0.58      0.66        98
        349       1.00      0.98      0.99       104
        350       0.76      0.85      0.80        86
        351       0.88      0.86      0.87       124
        352       0.43      0.59      0.50        63
        353       0.50      0.52      0.51        50
        354       0.48      0.70      0.57        40
        355       0.96      0.87      0.91       108
        356       0.82      0.89      0.86        47
        357       0.71      0.56      0.63        98
        358       0.64      0.81      0.72        42
        359       0.71      0.82      0.76       120
        360       1.00      0.90      0.95       102
        361       0.47      0.45      0.46       120
        362       0.94      0.93      0.94       157
        363       0.91      0.98      0.94        51
        364       0.43      0.67      0.52        42
        365       0.92      0.89      0.90       122
        366       0.71      0.60      0.65       109
        367       1.00      0.97      0.98        67
        368       0.82      0.83      0.83       113
        369       0.55      0.71      0.62        62
        370       0.96      0.92      0.94       130
        371       0.58      0.69      0.63        61
        372       0.78      0.83      0.81       109
        373       0.81      0.79      0.80       145
        374       0.46      0.71      0.56        31
        375       0.54      0.36      0.43        99
        376       0.97      0.92      0.95        38
        377       0.74      0.78      0.76        37
        378       0.22      0.58      0.32        31
        379       0.76      0.81      0.79        54
        380       0.74      0.91      0.82        44
        381       0.24      0.38      0.29        48
        382       0.74      0.90      0.81        31
        383       0.66      0.71      0.68        89
        384       0.99      1.00      1.00       100
        385       0.98      0.89      0.93        45
        386       0.82      0.85      0.84        60
        387       0.63      0.77      0.70        31
        388       0.67      0.70      0.69       115
        389       0.80      0.86      0.83        43
        390       0.99      0.99      0.99       158
        391       0.62      0.56      0.59        89
        392       0.82      0.92      0.87        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.71      0.75      0.73        80
        396       0.61      0.74      0.67        58
        397       0.96      0.93      0.94       121
        398       0.91      0.93      0.92        80
        399       0.95      0.93      0.94        44
        400       0.44      0.68      0.53        37
        401       0.84      0.75      0.79       108
        402       0.94      0.88      0.91        33
        403       0.93      0.88      0.90       123
        404       0.87      0.95      0.90        55
        405       0.90      0.71      0.79       102
        406       0.81      0.84      0.82       110
        407       0.63      0.62      0.62        76
        408       0.62      0.93      0.75        30
        409       0.48      0.60      0.53        42
        410       0.77      0.39      0.52        62
        411       0.73      0.69      0.71       135
        412       0.45      0.50      0.47        64
        413       0.97      1.00      0.98        31
        414       0.95      0.92      0.94        66
        415       0.28      0.25      0.26        88
        416       0.86      0.79      0.82       115
        417       0.97      0.95      0.96       124
        418       0.91      0.77      0.83       126
        419       0.78      0.48      0.60        75
        420       0.90      0.92      0.91        48
        421       0.88      1.00      0.94        30
        422       0.49      0.81      0.61        42
        423       0.64      0.67      0.65       109
        424       0.88      1.00      0.94        30
        425       0.80      0.89      0.85        37
        426       0.43      0.65      0.52        31
        427       0.69      0.97      0.81        32
        428       0.50      0.69      0.58        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       1.00      0.98      0.99        42
        432       0.65      0.67      0.66        70
        433       0.37      0.49      0.42        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.54      0.78      0.64        64
        437       0.72      0.88      0.79        32
        438       0.84      0.68      0.75        53
        439       0.70      0.70      0.70        30
        440       0.96      0.96      0.96        85
        441       0.34      0.64      0.44        33
        442       0.79      0.79      0.79       131
        443       0.53      0.60      0.56        85
        444       0.56      0.70      0.62        47
        445       0.85      0.87      0.86        38
        446       0.96      0.84      0.90       122
        447       0.62      0.56      0.59        63
        448       0.91      0.93      0.92        57
        449       0.89      0.98      0.93        57
        450       0.54      0.68      0.60        56
        451       0.65      0.69      0.67        51
        452       0.47      0.50      0.48        40
        453       0.95      0.91      0.93       137
        454       0.95      0.92      0.94        39
        455       0.73      0.87      0.80        63
        456       0.91      0.93      0.92        84
        457       0.91      0.78      0.84       136
        458       0.58      0.96      0.73        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.58        46
        462       0.51      0.47      0.49        68
        463       0.77      0.79      0.78       120
        464       0.87      0.83      0.85        93
        465       0.62      0.82      0.71        62
        466       0.46      0.77      0.58        31
        467       0.75      0.68      0.72        72
        468       0.89      0.98      0.93        41
        469       0.97      0.88      0.93        78
        470       0.97      0.92      0.94        36
        471       0.33      0.05      0.08        44
        472       0.64      0.72      0.68       112
        473       0.92      0.94      0.93        35
        474       0.53      0.55      0.54        78
        475       0.58      0.87      0.69        30
        476       0.91      0.86      0.88        78
        477       0.88      0.78      0.83       106
        478       0.71      0.47      0.57        32
        479       0.77      0.90      0.83        63
        480       0.94      0.98      0.96        91
        481       0.79      0.84      0.82        45
        482       0.86      0.94      0.90        32
        483       0.99      0.92      0.95       166
        484       0.94      1.00      0.97        34
        485       0.62      0.91      0.74        35
        486       0.32      0.33      0.33        73
        487       0.63      0.66      0.64        59
        488       0.45      0.81      0.57        31
        489       0.79      0.94      0.86        48
        490       0.63      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.79      0.81       114
        493       0.98      0.86      0.91       149
        494       0.67      0.67      0.67        51
        495       0.89      0.88      0.88        80
        496       0.93      1.00      0.96        38
        497       0.48      0.51      0.49        88

avg / total       0.77      0.76      0.76     35492

Iteration number, batch number :  4 35
Iteration number, batch number :  4 34
Iteration number, batch number :  4 33
Iteration number, batch number :  4 32
Iteration number, batch number :  4 31
Iteration number, batch number :  4 30
Iteration number, batch number :  4 29
Iteration number, batch number :  4 28
Iteration number, batch number :  4 27
Iteration number, batch number :  4 26
Iteration number, batch number :  4 25
Iteration number, batch number :  4 24
Iteration number, batch number :  4 23
Iteration number, batch number :  4 22
Iteration number, batch number :  4 21
Iteration number, batch number :  4 20
Iteration number, batch number :  4 19
Iteration number, batch number :  4 18
Iteration number, batch number :  4 17
Iteration number, batch number :  4 16
Iteration number, batch number :  4 15
Iteration number, batch number :  4 14
Iteration number, batch number :  4 13
Iteration number, batch number :  4 12
Iteration number, batch number :  4 11
Iteration number, batch number :  4 10
Iteration number, batch number :  4 9
Iteration number, batch number :  4 8
Iteration number, batch number :  4 7
Iteration number, batch number :  4 6
Iteration number, batch number :  4 5
Iteration number, batch number :  4 4
Iteration number, batch number :  4 3
Iteration number, batch number :  4 2
Iteration number, batch number :  4 1
Iteration number, batch number :  4 0
Accuracy on cv data :  79.1557097298 28295.0 28295
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.75      0.90      0.82        42
          2       0.64      0.92      0.76        37
          3       0.96      1.00      0.98        53
          4       0.97      0.85      0.91       131
          5       0.87      0.92      0.89        37
          6       0.45      0.67      0.54        51
          7       0.63      0.55      0.59        47
          8       0.63      0.55      0.58        77
          9       0.98      1.00      0.99        51
         10       0.96      0.93      0.95        46
         11       0.78      0.88      0.82        32
         12       0.86      0.94      0.90        79
         13       0.77      0.89      0.83        85
         14       0.93      0.97      0.95       143
         15       0.47      0.36      0.40       121
         16       0.85      0.94      0.89        35
         17       0.83      0.74      0.78       125
         18       0.84      0.80      0.82        70
         19       0.65      0.76      0.70        92
         20       0.67      0.74      0.71        39
         21       0.96      0.94      0.95        53
         22       0.69      0.72      0.70        46
         23       0.59      0.91      0.71        33
         24       0.73      0.90      0.81        40
         25       0.98      0.99      0.98        86
         26       0.96      0.90      0.93       132
         27       0.82      0.83      0.83        90
         28       0.54      0.84      0.66        31
         29       0.89      0.85      0.87        96
         30       0.86      0.88      0.87        74
         31       0.97      0.96      0.97        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.58      0.78      0.67        32
         35       0.88      0.90      0.89        77
         36       0.91      0.97      0.94        66
         37       0.77      0.77      0.77       115
         38       0.16      0.04      0.07       121
         39       0.91      0.91      0.91        35
         40       0.37      0.41      0.39        76
         41       0.89      0.83      0.86        93
         42       0.65      0.83      0.73        59
         43       0.91      0.95      0.93        63
         44       0.68      0.84      0.75        31
         45       0.82      0.93      0.87        40
         46       0.86      0.61      0.71       161
         47       0.60      0.75      0.67        48
         48       0.91      0.84      0.87        37
         49       0.76      0.89      0.82        38
         50       0.84      0.73      0.78       124
         51       0.53      0.41      0.46       101
         52       0.96      0.96      0.96        51
         53       0.96      0.99      0.97        77
         54       0.94      0.94      0.94        71
         55       1.00      0.95      0.97        95
         56       0.58      0.53      0.56        66
         57       0.80      0.75      0.77        80
         58       0.97      0.99      0.98        99
         59       0.98      0.90      0.94        93
         60       0.97      0.94      0.96        34
         61       0.87      1.00      0.93        41
         62       0.74      0.84      0.79        31
         63       0.99      0.91      0.94        74
         64       0.82      0.97      0.89        38
         65       0.92      0.88      0.90        96
         66       0.80      0.87      0.83        95
         67       0.86      0.90      0.88       132
         68       0.54      0.55      0.54        73
         69       0.41      0.40      0.41        30
         70       0.78      0.88      0.83        43
         71       0.62      0.59      0.60        44
         72       0.46      0.62      0.53        42
         73       0.94      0.84      0.89        37
         74       0.70      0.86      0.77        99
         75       0.88      0.87      0.88       164
         76       0.93      0.95      0.94        65
         77       0.96      0.96      0.96        71
         78       0.93      0.76      0.84        34
         79       0.70      0.85      0.77        33
         80       0.75      0.85      0.80        72
         81       0.92      0.98      0.95       132
         82       0.85      0.63      0.73       175
         83       0.78      0.73      0.75        48
         84       0.43      0.44      0.43        45
         85       0.98      0.96      0.97       110
         86       0.65      0.54      0.59        98
         87       0.79      0.73      0.76        83
         88       0.84      0.88      0.86        43
         89       0.84      0.89      0.86        35
         90       0.82      0.82      0.82        38
         91       0.66      0.53      0.59        51
         92       0.32      0.49      0.38        41
         93       0.88      0.92      0.90        48
         94       0.96      0.92      0.94        71
         95       0.82      0.81      0.82       117
         96       0.64      0.89      0.75        38
         97       0.67      0.84      0.74        31
         98       0.93      0.86      0.89        99
         99       0.78      0.75      0.76        69
        100       0.90      0.90      0.90        99
        101       0.66      0.61      0.63        85
        102       0.49      0.61      0.54        56
        103       0.94      0.94      0.94        36
        104       0.69      0.76      0.72        41
        105       0.70      0.65      0.67        65
        106       0.74      0.81      0.78        32
        107       0.93      0.74      0.83        35
        108       0.68      0.72      0.70        82
        109       0.91      0.81      0.86        78
        110       0.98      0.96      0.97       121
        111       0.66      0.76      0.71        54
        112       0.96      0.97      0.96        89
        113       0.72      0.94      0.82        31
        114       0.85      0.87      0.86       127
        115       0.96      0.94      0.95        78
        116       0.60      0.48      0.53       108
        117       0.85      0.87      0.86       110
        118       0.53      0.41      0.46        88
        119       0.89      0.77      0.82       107
        120       0.57      0.61      0.59        70
        121       0.77      0.95      0.85        39
        122       0.95      0.84      0.89       124
        123       0.85      0.76      0.80        59
        124       0.93      0.96      0.94        70
        125       0.78      0.92      0.85        39
        126       0.79      0.72      0.75       114
        127       1.00      1.00      1.00        56
        128       0.68      0.86      0.76        83
        129       0.97      0.91      0.94       125
        130       0.77      0.83      0.80        48
        131       0.85      0.88      0.86       112
        132       1.00      0.98      0.99        97
        133       0.99      0.74      0.85       164
        134       0.42      0.56      0.48        32
        135       0.82      0.88      0.85        81
        136       0.74      0.73      0.73       104
        137       0.70      0.69      0.70       134
        138       0.99      0.93      0.96        88
        139       0.31      0.13      0.19        38
        140       0.98      0.91      0.94        44
        141       0.90      0.84      0.87        62
        142       1.00      0.92      0.96       104
        143       0.62      0.71      0.66        51
        144       0.53      0.60      0.56        73
        145       0.70      0.88      0.78        50
        146       0.85      0.80      0.83        86
        147       0.88      0.88      0.88        93
        148       0.94      0.78      0.85        58
        149       0.92      0.90      0.91       110
        150       0.78      0.79      0.79        77
        151       0.66      0.64      0.65        75
        152       0.93      0.96      0.95        54
        153       0.47      0.94      0.62        32
        154       0.79      0.77      0.78        74
        155       0.88      0.71      0.79       189
        156       0.62      0.46      0.52        92
        157       0.88      0.94      0.91        31
        158       0.78      0.71      0.75       101
        159       0.54      0.57      0.55       109
        160       0.17      0.23      0.19        39
        161       0.66      0.74      0.69        57
        162       0.89      0.97      0.93        33
        163       0.40      0.36      0.38        70
        164       0.97      0.86      0.91       160
        165       0.36      0.38      0.37        40
        166       0.88      0.93      0.91        72
        167       0.76      0.84      0.80        64
        168       0.86      0.75      0.80        76
        169       0.76      0.85      0.80        40
        170       0.99      0.95      0.97       141
        171       0.17      0.06      0.09        62
        172       0.86      0.88      0.87        89
        173       0.65      0.72      0.68        39
        174       0.74      0.92      0.82        38
        175       0.78      0.96      0.86        45
        176       0.38      0.58      0.46        43
        177       0.67      0.88      0.76        32
        178       0.94      0.91      0.93        55
        179       0.42      0.53      0.47        32
        180       0.65      0.63      0.64        82
        181       0.63      0.59      0.61        32
        182       0.73      0.68      0.70        87
        183       0.65      0.86      0.74        42
        184       0.88      0.90      0.89        58
        185       0.74      0.87      0.80        30
        186       0.96      0.95      0.95        56
        187       0.68      0.97      0.80        31
        188       0.97      1.00      0.99        33
        189       0.78      0.91      0.84        47
        190       0.95      0.94      0.94       108
        191       0.98      1.00      0.99        40
        192       0.94      0.94      0.94       119
        193       0.94      1.00      0.97        33
        194       0.77      0.81      0.79       152
        195       0.94      0.83      0.88        71
        196       0.95      0.92      0.94        39
        197       0.94      0.94      0.94       124
        198       0.89      0.77      0.83        31
        199       0.43      0.55      0.48        38
        200       0.94      0.97      0.96        34
        201       0.32      0.35      0.34        31
        202       0.80      0.85      0.83        53
        203       0.86      0.91      0.89        34
        204       0.58      0.77      0.66        39
        205       0.63      0.71      0.67        75
        206       0.98      0.89      0.93       101
        207       0.35      0.36      0.35        45
        208       0.82      0.71      0.76       108
        209       0.90      0.94      0.92        80
        210       0.76      0.85      0.80        33
        211       0.86      0.95      0.90        60
        212       0.73      0.66      0.69        93
        213       0.80      0.77      0.78        87
        214       0.21      0.17      0.18        36
        215       0.47      0.47      0.47        92
        216       0.76      0.83      0.79        75
        217       0.54      0.58      0.56        71
        218       0.50      0.61      0.55        56
        219       0.87      0.77      0.81        60
        220       0.81      0.86      0.83        50
        221       0.57      0.65      0.61        43
        222       1.00      1.00      1.00        46
        223       0.72      0.71      0.72        97
        224       1.00      0.84      0.91        85
        225       0.44      0.73      0.55        33
        226       0.91      1.00      0.96        43
        227       0.98      0.94      0.96        54
        228       0.52      0.80      0.63        30
        229       0.37      0.69      0.48        32
        230       0.57      0.57      0.57        37
        231       0.94      0.81      0.87        59
        232       0.92      0.86      0.89        98
        233       0.94      0.94      0.94        70
        234       0.96      0.97      0.96        69
        235       0.87      0.86      0.86       104
        236       1.00      0.96      0.98        99
        237       0.93      0.86      0.89       123
        238       0.89      0.89      0.89        93
        239       1.00      0.96      0.98        48
        240       0.57      0.62      0.60        82
        241       0.76      0.85      0.80        33
        242       0.83      0.85      0.84        95
        243       0.82      0.89      0.85        63
        244       0.41      0.54      0.46        41
        245       0.78      0.70      0.74       104
        246       0.86      0.70      0.77        77
        247       0.95      1.00      0.98        62
        248       0.97      0.95      0.96        78
        249       0.88      0.74      0.80        61
        250       0.92      0.83      0.87       121
        251       0.63      0.55      0.59        31
        252       0.72      0.93      0.81        42
        253       0.58      0.64      0.61        45
        254       0.22      0.29      0.25        48
        255       1.00      1.00      1.00        42
        256       1.00      0.98      0.99        98
        257       0.89      0.97      0.93        32
        258       0.46      0.53      0.49        55
        259       0.92      0.95      0.93       128
        260       0.62      0.48      0.54       121
        261       0.57      0.85      0.68        39
        262       0.98      1.00      0.99        62
        263       0.49      0.67      0.57        64
        264       0.93      0.93      0.93        45
        265       0.87      0.96      0.92        56
        266       0.90      0.94      0.92        68
        267       0.87      0.87      0.87        67
        268       0.84      0.82      0.83       148
        269       0.60      0.81      0.68        31
        270       0.66      0.42      0.51       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.96      1.00      0.98        44
        274       0.79      0.77      0.78       168
        275       0.31      0.63      0.42        38
        276       0.26      0.51      0.34        35
        277       0.87      0.93      0.90        58
        278       0.69      0.76      0.72        33
        279       0.51      0.57      0.54        63
        280       0.95      0.88      0.91        67
        281       0.66      0.82      0.73        45
        282       0.78      0.94      0.85        49
        283       0.82      1.00      0.90        32
        284       0.72      0.60      0.66        96
        285       0.76      0.80      0.78       118
        286       0.84      0.86      0.85        66
        287       0.72      0.67      0.70       107
        288       0.92      0.94      0.93        35
        289       0.93      0.86      0.89        88
        290       0.81      0.83      0.82        66
        291       0.99      0.88      0.93        96
        292       0.99      0.81      0.89       135
        293       0.73      0.84      0.78        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.63      0.59      0.61        82
        297       0.90      0.83      0.86       107
        298       0.73      0.78      0.75        41
        299       0.86      0.81      0.83        31
        300       0.79      0.88      0.83        42
        301       0.73      0.83      0.78        53
        302       0.97      0.93      0.95        90
        303       0.57      0.77      0.66        35
        304       0.91      0.98      0.94        42
        305       0.93      0.89      0.91       102
        306       1.00      0.95      0.97        39
        307       0.87      0.75      0.81       120
        308       0.96      0.99      0.97       117
        309       0.91      0.87      0.89        98
        310       0.96      0.99      0.98       110
        311       0.44      0.44      0.44        36
        312       0.96      1.00      0.98        99
        313       0.79      0.94      0.86       116
        314       0.60      0.74      0.67        39
        315       0.62      0.70      0.66        30
        316       0.93      0.83      0.88       156
        317       0.76      0.84      0.80        75
        318       0.64      0.71      0.67        72
        319       0.98      0.90      0.94       154
        320       0.95      0.85      0.90        86
        321       0.28      0.17      0.21        59
        322       0.75      0.80      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.88      0.91        56
        325       0.56      0.56      0.56       112
        326       0.91      0.90      0.91       114
        327       0.87      0.85      0.86        93
        328       0.75      0.71      0.73        85
        329       0.29      0.14      0.19        76
        330       0.61      0.67      0.64        66
        331       0.88      0.90      0.89       128
        332       0.88      0.91      0.90        33
        333       0.49      0.55      0.52        56
        334       0.62      0.44      0.51        41
        335       0.86      0.65      0.74       103
        336       0.76      0.83      0.79        46
        337       0.96      0.93      0.94        54
        338       0.77      0.92      0.84        52
        339       0.70      0.68      0.69        57
        340       0.84      0.76      0.80       128
        341       0.91      0.93      0.92        42
        342       0.80      0.64      0.71       118
        343       0.55      0.58      0.57        45
        344       0.56      0.71      0.63        38
        345       0.99      0.97      0.98       153
        346       0.60      0.78      0.68        65
        347       0.91      0.83      0.87        64
        348       0.75      0.56      0.64        99
        349       1.00      0.99      1.00       105
        350       0.78      0.80      0.79        86
        351       0.94      0.95      0.94       125
        352       0.43      0.38      0.40        64
        353       0.40      0.56      0.47        50
        354       0.39      0.68      0.50        41
        355       0.94      0.85      0.89       108
        356       0.80      0.75      0.77        48
        357       0.68      0.65      0.67        98
        358       0.65      0.65      0.65        43
        359       0.71      0.74      0.73       120
        360       1.00      0.98      0.99       103
        361       0.41      0.42      0.42       120
        362       0.98      0.89      0.93       158
        363       0.93      0.98      0.95        51
        364       0.55      0.76      0.64        42
        365       0.88      0.86      0.87       122
        366       0.73      0.58      0.65       110
        367       1.00      0.94      0.97        67
        368       0.84      0.86      0.85       114
        369       0.67      0.82      0.74        62
        370       0.92      0.93      0.92       130
        371       0.67      0.69      0.68        62
        372       0.80      0.83      0.81       110
        373       0.89      0.86      0.87       145
        374       0.48      0.75      0.59        32
        375       0.48      0.38      0.42       100
        376       0.97      0.95      0.96        39
        377       0.67      0.74      0.70        38
        378       0.36      0.55      0.44        31
        379       0.74      0.83      0.78        54
        380       0.72      0.89      0.80        44
        381       0.31      0.35      0.33        48
        382       0.70      0.84      0.76        31
        383       0.71      0.79      0.74        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.75      0.80      0.78        61
        387       0.60      0.75      0.67        32
        388       0.69      0.67      0.68       115
        389       0.76      0.86      0.80        43
        390       0.99      0.97      0.98       158
        391       0.64      0.72      0.68        89
        392       0.84      0.96      0.90        50
        393       0.97      0.93      0.95        69
        394       0.84      0.87      0.86        55
        395       0.80      0.85      0.83        81
        396       0.72      0.75      0.73        59
        397       0.94      0.97      0.95       121
        398       0.89      0.94      0.91        80
        399       0.83      0.87      0.85        45
        400       0.54      0.50      0.52        38
        401       0.76      0.84      0.80       108
        402       0.91      0.91      0.91        34
        403       0.89      0.84      0.86       123
        404       0.83      0.89      0.86        55
        405       0.89      0.83      0.86       103
        406       0.76      0.87      0.82       111
        407       0.72      0.76      0.74        76
        408       0.60      0.87      0.71        30
        409       0.57      0.60      0.58        42
        410       0.78      0.46      0.58        63
        411       0.82      0.74      0.78       136
        412       0.60      0.52      0.55        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.36      0.37        88
        416       0.90      0.71      0.79       116
        417       0.94      0.94      0.94       124
        418       0.90      0.72      0.80       127
        419       0.83      0.89      0.86        76
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.61      0.81      0.69        42
        423       0.76      0.71      0.73       110
        424       0.69      0.97      0.81        30
        425       0.75      0.89      0.81        37
        426       0.69      0.78      0.74        32
        427       0.73      0.94      0.82        32
        428       0.46      0.65      0.54        55
        429       0.96      0.98      0.97        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.75      0.65      0.70        71
        433       0.48      0.49      0.49        55
        434       0.88      0.94      0.91        32
        435       0.92      0.82      0.86        66
        436       0.90      0.86      0.88        65
        437       0.70      0.94      0.80        32
        438       0.74      0.65      0.69        54
        439       0.78      0.70      0.74        30
        440       0.99      0.97      0.98        86
        441       0.38      0.61      0.47        33
        442       0.81      0.77      0.79       131
        443       0.51      0.63      0.57        86
        444       0.51      0.46      0.48        48
        445       0.75      0.95      0.84        38
        446       0.97      0.98      0.97       123
        447       0.61      0.75      0.67        63
        448       0.87      0.78      0.82        58
        449       0.89      0.95      0.92        57
        450       0.58      0.73      0.65        56
        451       0.70      0.76      0.73        51
        452       0.52      0.71      0.60        41
        453       0.96      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.79      0.87      0.83        63
        456       0.93      0.80      0.86        85
        457       0.91      0.78      0.84       137
        458       0.76      0.81      0.78        47
        459       0.60      0.38      0.46        87
        460       0.99      0.99      0.99        95
        461       0.52      0.74      0.61        46
        462       0.57      0.51      0.54        69
        463       0.89      0.71      0.79       120
        464       0.87      0.83      0.85        93
        465       0.71      0.87      0.78        63
        466       0.38      0.53      0.44        32
        467       0.76      0.70      0.73        73
        468       0.93      0.98      0.95        42
        469       0.91      0.87      0.89        78
        470       1.00      0.97      0.99        36
        471       0.78      0.66      0.72        44
        472       0.68      0.64      0.66       112
        473       0.87      0.94      0.91        36
        474       0.54      0.46      0.49        79
        475       0.47      0.74      0.58        31
        476       0.97      0.90      0.93        78
        477       0.93      0.82      0.87       107
        478       0.95      0.58      0.72        33
        479       0.88      0.94      0.91        63
        480       0.91      0.95      0.93        91
        481       0.71      0.76      0.74        46
        482       0.97      0.88      0.92        33
        483       0.99      0.86      0.92       167
        484       1.00      0.97      0.99        35
        485       0.82      0.75      0.78        36
        486       0.28      0.30      0.29        74
        487       0.72      0.78      0.75        60
        488       0.55      0.68      0.61        31
        489       0.92      0.92      0.92        49
        490       0.82      0.63      0.71       121
        491       0.82      0.85      0.84        39
        492       0.73      0.73      0.73       114
        493       0.95      0.85      0.89       149
        494       0.64      0.71      0.67        51
        495       0.91      0.93      0.92        80
        496       0.81      0.97      0.88        39
        497       0.57      0.44      0.50        88

avg / total       0.80      0.79      0.79     35746

Iteration number, batch number :  5 0
Training data accuracy :  0.83734939759
Training data loss     :  0.00532917199687
Iteration number, batch number :  5 1
Training data accuracy :  0.822289156627
Training data loss     :  0.00541339073786
Iteration number, batch number :  5 2
Training data accuracy :  0.827309236948
Training data loss     :  0.00537313659754
Iteration number, batch number :  5 3
Training data accuracy :  0.836345381526
Training data loss     :  0.00538046758921
Iteration number, batch number :  5 4
Training data accuracy :  0.838353413655
Training data loss     :  0.00534612299963
Iteration number, batch number :  5 5
Training data accuracy :  0.857429718876
Training data loss     :  0.00515275731365
Iteration number, batch number :  5 6
Training data accuracy :  0.838353413655
Training data loss     :  0.00554711532869
Iteration number, batch number :  5 7
Training data accuracy :  0.852409638554
Training data loss     :  0.0047421571621
Iteration number, batch number :  5 8
Training data accuracy :  0.831325301205
Training data loss     :  0.00517956606981
Iteration number, batch number :  5 9
Training data accuracy :  0.843373493976
Training data loss     :  0.00524066108079
Iteration number, batch number :  5 10
Training data accuracy :  0.863453815261
Training data loss     :  0.00467910216001
Iteration number, batch number :  5 11
Training data accuracy :  0.838353413655
Training data loss     :  0.00508516996098
Iteration number, batch number :  5 12
Training data accuracy :  0.856425702811
Training data loss     :  0.00502304374356
Iteration number, batch number :  5 13
Training data accuracy :  0.833333333333
Training data loss     :  0.0054855749113
Iteration number, batch number :  5 14
Training data accuracy :  0.848393574297
Training data loss     :  0.00483186416492
Iteration number, batch number :  5 15
Training data accuracy :  0.85140562249
Training data loss     :  0.00474595140631
Iteration number, batch number :  5 16
Training data accuracy :  0.872489959839
Training data loss     :  0.00444870418262
Iteration number, batch number :  5 17
Training data accuracy :  0.843373493976
Training data loss     :  0.0047374455846
Iteration number, batch number :  5 18
Training data accuracy :  0.846385542169
Training data loss     :  0.00513404439888
Iteration number, batch number :  5 19
Training data accuracy :  0.85843373494
Training data loss     :  0.00494866676225
Iteration number, batch number :  5 20
Training data accuracy :  0.868473895582
Training data loss     :  0.00465994201481
Iteration number, batch number :  5 21
Training data accuracy :  0.839357429719
Training data loss     :  0.00476514854447
Iteration number, batch number :  5 22
Training data accuracy :  0.853413654618
Training data loss     :  0.00451346064639
Iteration number, batch number :  5 23
Training data accuracy :  0.848393574297
Training data loss     :  0.00449507849687
Iteration number, batch number :  5 24
Training data accuracy :  0.847389558233
Training data loss     :  0.00495481536491
Iteration number, batch number :  5 25
Training data accuracy :  0.848393574297
Training data loss     :  0.00502799753466
Iteration number, batch number :  5 26
Training data accuracy :  0.862449799197
Training data loss     :  0.00473999938801
Iteration number, batch number :  5 27
Training data accuracy :  0.85140562249
Training data loss     :  0.00481115631926
Iteration number, batch number :  5 28
Training data accuracy :  0.874497991968
Training data loss     :  0.00439797182507
Iteration number, batch number :  5 29
Training data accuracy :  0.847389558233
Training data loss     :  0.00487918888547
Iteration number, batch number :  5 30
Training data accuracy :  0.849397590361
Training data loss     :  0.00471808885316
Iteration number, batch number :  5 31
Training data accuracy :  0.846385542169
Training data loss     :  0.00497036814256
Iteration number, batch number :  5 32
Training data accuracy :  0.852409638554
Training data loss     :  0.00485135625697
Iteration number, batch number :  5 33
Training data accuracy :  0.88453815261
Training data loss     :  0.00445396879197
Iteration number, batch number :  5 34
Training data accuracy :  0.878514056225
Training data loss     :  0.0043158195751
Iteration number, batch number :  5 35
Training data accuracy :  0.869477911647
Training data loss     :  0.00417934898699
Iteration number, batch number :  5 36
Training data accuracy :  0.85843373494
Training data loss     :  0.00441045367112
Iteration number, batch number :  5 37
Training data accuracy :  0.873493975904
Training data loss     :  0.00448343135181
Iteration number, batch number :  5 38
Training data accuracy :  0.868473895582
Training data loss     :  0.00451641336974
Iteration number, batch number :  5 39
Training data accuracy :  0.890562248996
Training data loss     :  0.00396785354871
Iteration number, batch number :  5 40
Training data accuracy :  0.855421686747
Training data loss     :  0.00486508086239
Iteration number, batch number :  5 41
Training data accuracy :  0.860441767068
Training data loss     :  0.00446038622196
Iteration number, batch number :  5 42
Training data accuracy :  0.875502008032
Training data loss     :  0.00442273322044
Iteration number, batch number :  5 43
Training data accuracy :  0.873493975904
Training data loss     :  0.00443198259956
Iteration number, batch number :  5 44
Training data accuracy :  0.871485943775
Training data loss     :  0.00405521168806
Iteration number, batch number :  5 45
Training data accuracy :  0.873493975904
Training data loss     :  0.0042715998831
Iteration number, batch number :  5 46
Training data accuracy :  0.867469879518
Training data loss     :  0.0042371764725
Iteration number, batch number :  5 47
Training data accuracy :  0.877510040161
Training data loss     :  0.00457490235095
Iteration number, batch number :  5 48
Training data accuracy :  0.856425702811
Training data loss     :  0.00474095130602
Iteration number, batch number :  5 49
Training data accuracy :  0.856425702811
Training data loss     :  0.00442923094724
Iteration number, batch number :  5 50
Training data accuracy :  0.869477911647
Training data loss     :  0.00446824929858
Iteration number, batch number :  5 51
Training data accuracy :  0.842369477912
Training data loss     :  0.00484425342242
Iteration number, batch number :  5 52
Training data accuracy :  0.847389558233
Training data loss     :  0.00533620292928
Iteration number, batch number :  5 53
^[[A^[[Training data accuracy :  0.845381526104
Training data loss     :  0.00505011015904
Iteration number, batch number :  5 54
Training data accuracy :  0.849397590361
Training data loss     :  0.00501780764724
Iteration number, batch number :  5 55
Training data accuracy :  0.841365461847
Training data loss     :  0.00485965893383
Iteration number, batch number :  5 56
Training data accuracy :  0.856425702811
Training data loss     :  0.00509168931311
Iteration number, batch number :  5 57
Training data accuracy :  0.83734939759
Training data loss     :  0.005201242408
Iteration number, batch number :  5 58
Training data accuracy :  0.826305220884
Training data loss     :  0.00519081376137
Iteration number, batch number :  5 59
Training data accuracy :  0.84437751004
Training data loss     :  0.00543640010274
Iteration number, batch number :  5 60
Training data accuracy :  0.84437751004
Training data loss     :  0.0051246140923
Iteration number, batch number :  5 61
Training data accuracy :  0.835341365462
Training data loss     :  0.00529691174723
Iteration number, batch number :  5 62
Training data accuracy :  0.836345381526
Training data loss     :  0.00523743329393
Iteration number, batch number :  5 63
Training data accuracy :  0.833333333333
Training data loss     :  0.0053665590584
Iteration number, batch number :  5 64
Training data accuracy :  0.845381526104
Training data loss     :  0.00561435636905
Iteration number, batch number :  5 65
Training data accuracy :  0.834337349398
Training data loss     :  0.00579117528503
Iteration number, batch number :  5 66
Training data accuracy :  0.840361445783
Training data loss     :  0.00513416093777
Iteration number, batch number :  5 67
Training data accuracy :  0.856425702811
Training data loss     :  0.00528023945273
Iteration number, batch number :  5 68
Training data accuracy :  0.840361445783
Training data loss     :  0.00551581161834
Iteration number, batch number :  5 69
Training data accuracy :  0.834337349398
Training data loss     :  0.00522859677018
Iteration number, batch number :  5 35
Iteration number, batch number :  5 34
Iteration number, batch number :  5 33
Iteration number, batch number :  5 32
Iteration number, batch number :  5 31
Iteration number, batch number :  5 30
Iteration number, batch number :  5 29
Iteration number, batch number :  5 28
Iteration number, batch number :  5 27
Iteration number, batch number :  5 26
Iteration number, batch number :  5 25
Iteration number, batch number :  5 24
Iteration number, batch number :  5 23
Iteration number, batch number :  5 22
Iteration number, batch number :  5 21
Iteration number, batch number :  5 20
Iteration number, batch number :  5 19
Iteration number, batch number :  5 18
Iteration number, batch number :  5 17
Iteration number, batch number :  5 16
Iteration number, batch number :  5 15
Iteration number, batch number :  5 14
Iteration number, batch number :  5 13
Iteration number, batch number :  5 12
Iteration number, batch number :  5 11
Iteration number, batch number :  5 10
Iteration number, batch number :  5 9
Iteration number, batch number :  5 8
Iteration number, batch number :  5 7
Iteration number, batch number :  5 6
Iteration number, batch number :  5 5
Iteration number, batch number :  5 4
Iteration number, batch number :  5 3
Iteration number, batch number :  5 2
Iteration number, batch number :  5 1
Iteration number, batch number :  5 0
Accuracy on test data :  76.2735264285 27071.0 27071
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.77      0.91      0.84        45
          1       0.74      0.95      0.83        41
          2       0.64      0.81      0.71        37
          3       0.83      0.96      0.89        52
          4       0.95      0.55      0.69       130
          5       0.92      0.97      0.95        36
          6       0.48      0.71      0.57        51
          7       0.71      0.62      0.66        47
          8       0.38      0.38      0.38        76
          9       0.98      1.00      0.99        51
         10       0.95      0.93      0.94        45
         11       0.35      0.45      0.39        31
         12       0.94      0.85      0.89        79
         13       0.71      0.82      0.76        85
         14       0.92      0.85      0.89       143
         15       0.52      0.42      0.46       120
         16       0.91      0.94      0.93        34
         17       0.65      0.31      0.42       124
         18       0.79      0.80      0.79        70
         19       0.52      0.55      0.53        91
         20       0.39      0.62      0.48        39
         21       0.91      0.38      0.53        53
         22       0.52      0.51      0.52        45
         23       0.49      0.73      0.59        33
         24       0.79      0.78      0.78        40
         25       0.98      0.98      0.98        85
         26       0.92      0.73      0.81       132
         27       0.92      0.88      0.90        90
         28       0.36      0.53      0.43        30
         29       0.85      0.84      0.85        96
         30       0.86      0.86      0.86        74
         31       0.97      0.89      0.93        79
         32       1.00      0.99      0.99       269
         33       0.97      0.94      0.95        31
         34       0.40      0.66      0.50        32
         35       0.87      0.95      0.91        77
         36       0.87      0.91      0.89        66
         37       0.75      0.83      0.79       114
         38       0.06      0.01      0.01       121
         39       0.80      0.82      0.81        34
         40       0.34      0.45      0.39        76
         41       0.91      0.64      0.75        92
         42       0.65      0.72      0.68        58
         43       0.95      0.87      0.91        63
         44       0.56      0.80      0.66        30
         45       0.83      0.95      0.88        40
         46       0.00      0.00      0.00       161
         47       0.58      0.77      0.66        47
         48       0.92      0.94      0.93        36
         49       0.81      0.95      0.88        37
         50       0.78      0.65      0.71       123
         51       0.60      0.47      0.53       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.94      0.90      0.92        70
         55       0.98      0.94      0.96        94
         56       0.59      0.64      0.61        66
         57       0.67      0.65      0.66        79
         58       0.97      0.92      0.94        98
         59       0.95      0.89      0.92        92
         60       1.00      0.88      0.94        33
         61       0.88      0.93      0.90        41
         62       0.33      0.07      0.11        30
         63       0.84      0.88      0.86        74
         64       0.86      1.00      0.93        38
         65       0.93      0.88      0.91        95
         66       0.79      0.68      0.73        94
         67       0.88      0.89      0.89       132
         68       0.39      0.43      0.41        72
         69       0.38      0.53      0.44        30
         70       0.74      0.81      0.77        42
         71       0.67      0.80      0.73        44
         72       0.33      0.67      0.44        42
         73       0.86      0.84      0.85        37
         74       0.64      0.89      0.74        98
         75       0.95      0.87      0.91       163
         76       0.91      0.91      0.91        65
         77       0.97      0.94      0.96        70
         78       0.85      0.85      0.85        33
         79       0.60      0.79      0.68        33
         80       0.76      0.66      0.71        71
         81       0.96      0.98      0.97       131
         82       0.83      0.69      0.75       174
         83       0.71      0.74      0.73        47
         84       0.24      0.30      0.26        44
         85       0.96      0.93      0.94       110
         86       0.53      0.45      0.49        97
         87       0.92      0.78      0.84        83
         88       0.77      0.84      0.80        43
         89       0.71      0.91      0.80        35
         90       0.72      0.82      0.77        38
         91       0.68      0.63      0.65        51
         92       0.44      0.72      0.55        40
         93       0.94      0.92      0.93        48
         94       0.89      0.90      0.90        71
         95       0.82      0.79      0.80       117
         96       0.55      0.84      0.67        38
         97       0.68      0.90      0.77        30
         98       0.88      0.85      0.86        98
         99       0.63      0.74      0.68        69
        100       0.92      0.91      0.91        98
        101       0.63      0.65      0.64        85
        102       0.37      0.55      0.44        56
        103       0.97      0.92      0.94        36
        104       0.69      0.60      0.64        40
        105       0.65      0.55      0.60        65
        106       0.62      0.72      0.67        32
        107       1.00      0.53      0.69        34
        108       0.69      0.75      0.72        81
        109       0.62      0.26      0.37        77
        110       0.97      0.92      0.94       121
        111       0.56      0.55      0.55        53
        112       0.91      0.89      0.90        88
        113       0.76      0.97      0.85        30
        114       0.84      0.88      0.86       127
        115       0.92      0.73      0.81        78
        116       0.58      0.51      0.54       108
        117       0.92      0.83      0.87       110
        118       0.51      0.41      0.46        88
        119       0.90      0.80      0.85       106
        120       0.62      0.77      0.68        69
        121       0.78      0.82      0.80        39
        122       0.87      0.77      0.82       124
        123       0.83      0.83      0.83        59
        124       1.00      0.96      0.98        69
        125       0.72      0.87      0.79        38
        126       0.76      0.64      0.69       113
        127       0.98      1.00      0.99        55
        128       0.69      0.85      0.77        82
        129       0.93      0.89      0.91       125
        130       0.57      0.83      0.67        47
        131       0.82      0.36      0.50       112
        132       0.99      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.47      0.59      0.53        32
        135       0.81      0.79      0.80        80
        136       0.72      0.71      0.72       103
        137       0.76      0.81      0.78       134
        138       0.94      0.91      0.92        88
        139       0.15      0.08      0.11        37
        140       0.92      1.00      0.96        44
        141       0.84      0.92      0.88        61
        142       0.98      0.94      0.96       103
        143       0.52      0.69      0.59        51
        144       0.47      0.65      0.55        72
        145       0.72      0.96      0.82        49
        146       0.76      0.74      0.75        85
        147       0.89      0.87      0.88        93
        148       0.86      0.86      0.86        57
        149       0.84      0.86      0.85       109
        150       0.91      0.64      0.75        76
        151       0.68      0.69      0.69        75
        152       0.94      0.91      0.92        54
        153       0.49      0.78      0.60        32
        154       0.90      0.88      0.89        73
        155       0.83      0.59      0.69       189
        156       0.63      0.48      0.54        92
        157       0.91      0.94      0.92        31
        158       0.50      0.15      0.23       100
        159       0.51      0.41      0.45       108
        160       0.24      0.44      0.31        39
        161       0.67      0.74      0.70        57
        162       0.92      1.00      0.96        33
        163       0.38      0.47      0.42        70
        164       0.96      0.81      0.88       159
        165       0.49      0.72      0.58        39
        166       0.92      0.94      0.93        71
        167       0.85      0.80      0.82        64
        168       0.85      0.81      0.83        75
        169       0.53      0.64      0.58        39
        170       0.99      0.94      0.96       140
        171       0.15      0.08      0.11        61
        172       0.79      0.90      0.84        88
        173       0.61      0.69      0.65        39
        174       0.82      0.89      0.86        37
        175       0.77      0.82      0.80        45
        176       0.36      0.50      0.42        42
        177       0.61      0.87      0.72        31
        178       0.93      0.95      0.94        55
        179       0.38      0.58      0.46        31
        180       0.63      0.62      0.63        82
        181       0.59      0.65      0.62        31
        182       0.67      0.67      0.67        87
        183       0.74      0.90      0.81        41
        184       0.87      0.91      0.89        57
        185       0.76      0.87      0.81        30
        186       1.00      0.95      0.97        56
        187       0.74      0.97      0.84        30
        188       0.97      1.00      0.99        33
        189       0.70      0.80      0.75        46
        190       0.91      0.99      0.95       107
        191       0.93      0.95      0.94        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.80      0.74      0.77       152
        195       0.91      0.85      0.88        71
        196       0.95      0.95      0.95        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.39      0.70      0.50        37
        200       0.94      1.00      0.97        34
        201       0.16      0.27      0.20        30
        202       0.74      0.81      0.77        53
        203       0.89      0.97      0.93        33
        204       0.66      0.92      0.77        38
        205       0.49      0.55      0.52        75
        206       0.95      0.92      0.93       101
        207       0.29      0.27      0.28        45
        208       0.79      0.77      0.78       108
        209       0.78      0.84      0.80        79
        210       0.70      0.94      0.80        32
        211       0.92      0.90      0.91        60
        212       0.73      0.70      0.71        93
        213       0.76      0.79      0.78        87
        214       0.23      0.34      0.28        35
        215       0.57      0.55      0.56        91
        216       0.75      0.53      0.62        75
        217       0.53      0.61      0.57        70
        218       0.58      0.69      0.63        55
        219       0.92      0.73      0.81        60
        220       0.79      0.92      0.85        50
        221       0.53      0.69      0.60        42
        222       1.00      0.96      0.98        45
        223       0.79      0.72      0.75        96
        224       0.92      0.85      0.88        85
        225       0.51      0.84      0.64        32
        226       0.95      0.93      0.94        43
        227       0.93      0.98      0.95        54
        228       0.42      0.83      0.56        30
        229       0.26      0.50      0.34        32
        230       0.50      0.57      0.53        37
        231       0.84      0.90      0.87        59
        232       0.96      0.84      0.90        97
        233       0.97      0.94      0.96        69
        234       0.90      0.55      0.68        69
        235       0.85      0.75      0.79       103
        236       0.99      0.98      0.98        98
        237       0.91      0.72      0.80       123
        238       0.86      0.89      0.87        93
        239       1.00      0.96      0.98        48
        240       0.61      0.65      0.63        81
        241       0.83      0.94      0.88        32
        242       0.88      0.74      0.80        95
        243       0.79      0.87      0.83        63
        244       0.40      0.57      0.47        40
        245       0.78      0.78      0.78       103
        246       0.71      0.51      0.59        77
        247       0.94      0.98      0.96        61
        248       0.96      0.90      0.93        78
        249       0.91      0.83      0.87        60
        250       0.94      0.93      0.94       120
        251       0.56      0.73      0.64        30
        252       0.82      0.86      0.84        42
        253       0.55      0.66      0.60        44
        254       0.24      0.32      0.28        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.81      0.94      0.87        31
        258       0.38      0.33      0.35        55
        259       0.96      0.91      0.94       128
        260       0.60      0.41      0.49       121
        261       0.63      0.85      0.73        39
        262       0.97      0.98      0.98        62
        263       0.51      0.70      0.59        63
        264       0.87      0.89      0.88        45
        265       0.71      0.91      0.80        55
        266       0.91      0.94      0.93        67
        267       0.86      0.77      0.82        66
        268       0.80      0.69      0.74       148
        269       0.49      0.90      0.64        30
        270       0.58      0.45      0.51       149
        271       0.81      0.93      0.87        42
        272       0.97      0.95      0.96       119
        273       0.88      1.00      0.93        43
        274       0.80      0.75      0.77       167
        275       0.33      0.46      0.39        37
        276       0.19      0.40      0.26        35
        277       0.93      0.96      0.95        57
        278       0.43      0.28      0.34        32
        279       0.47      0.52      0.49        62
        280       0.00      0.00      0.00        66
        281       0.65      0.87      0.74        45
        282       0.86      0.90      0.88        48
        283       0.79      0.97      0.87        31
        284       0.69      0.65      0.67        96
        285       0.77      0.75      0.76       117
        286       0.97      0.85      0.90        66
        287       0.74      0.71      0.72       107
        288       0.69      0.89      0.78        35
        289       0.96      0.92      0.94        87
        290       0.70      0.73      0.71        66
        291       0.96      0.89      0.92        96
        292       0.97      0.70      0.82       135
        293       0.61      0.91      0.73        43
        294       0.47      0.90      0.62        31
        295       0.77      0.84      0.80        56
        296       0.46      0.53      0.49        81
        297       0.89      0.89      0.89       107
        298       0.67      0.68      0.67        41
        299       0.66      0.87      0.75        31
        300       0.80      0.80      0.80        41
        301       0.69      0.81      0.74        52
        302       0.80      0.91      0.85        90
        303       0.46      0.82      0.59        34
        304       0.93      0.95      0.94        41
        305       0.96      0.86      0.91       101
        306       0.92      0.95      0.94        38
        307       0.90      0.67      0.77       120
        308       0.97      0.96      0.96       117
        309       0.91      0.86      0.88        97
        310       0.95      0.95      0.95       109
        311       0.38      0.46      0.42        35
        312       0.94      0.94      0.94        98
        313       0.66      0.91      0.77       116
        314       0.48      0.72      0.58        39
        315       0.76      0.83      0.79        30
        316       0.95      0.91      0.93       155
        317       0.78      0.84      0.81        75
        318       0.71      0.64      0.67        72
        319       0.98      0.86      0.91       154
        320       0.84      0.82      0.83        85
        321       0.04      0.02      0.02        58
        322       0.76      0.81      0.78       119
        323       1.00      1.00      1.00        43
        324       0.93      0.96      0.95        55
        325       0.60      0.59      0.59       112
        326       0.90      0.87      0.88       114
        327       0.84      0.79      0.82        92
        328       0.67      0.69      0.68        84
        329       0.30      0.15      0.20        75
        330       0.66      0.67      0.66        66
        331       0.94      0.86      0.90       127
        332       0.81      0.88      0.84        33
        333       0.48      0.56      0.52        55
        334       0.52      0.29      0.38        41
        335       0.80      0.78      0.79       103
        336       0.73      0.89      0.80        45
        337       0.91      0.91      0.91        54
        338       0.79      0.88      0.83        51
        339       0.69      0.75      0.72        57
        340       0.76      0.76      0.76       127
        341       0.93      0.95      0.94        41
        342       0.74      0.69      0.71       118
        343       0.61      0.67      0.64        45
        344       0.62      0.89      0.73        37
        345       0.99      0.97      0.98       152
        346       0.59      0.69      0.63        64
        347       0.88      0.92      0.90        63
        348       0.73      0.58      0.65        98
        349       1.00      0.98      0.99       104
        350       0.77      0.85      0.81        86
        351       0.88      0.86      0.87       124
        352       0.44      0.59      0.50        63
        353       0.50      0.52      0.51        50
        354       0.49      0.70      0.58        40
        355       0.95      0.87      0.91       108
        356       0.84      0.89      0.87        47
        357       0.71      0.57      0.63        98
        358       0.65      0.81      0.72        42
        359       0.74      0.82      0.78       120
        360       1.00      0.92      0.96       102
        361       0.48      0.45      0.47       120
        362       0.94      0.93      0.94       157
        363       0.91      0.98      0.94        51
        364       0.44      0.67      0.53        42
        365       0.92      0.89      0.90       122
        366       0.71      0.61      0.65       109
        367       1.00      0.97      0.98        67
        368       0.83      0.85      0.84       113
        369       0.55      0.71      0.62        62
        370       0.98      0.92      0.95       130
        371       0.56      0.69      0.62        61
        372       0.80      0.83      0.82       109
        373       0.82      0.80      0.81       145
        374       0.46      0.71      0.56        31
        375       0.52      0.34      0.41        99
        376       0.97      0.92      0.95        38
        377       0.74      0.76      0.75        37
        378       0.23      0.58      0.32        31
        379       0.77      0.81      0.79        54
        380       0.74      0.91      0.82        44
        381       0.23      0.38      0.28        48
        382       0.68      0.90      0.78        31
        383       0.66      0.73      0.69        89
        384       0.99      1.00      1.00       100
        385       0.98      0.89      0.93        45
        386       0.82      0.83      0.83        60
        387       0.62      0.77      0.69        31
        388       0.68      0.70      0.69       115
        389       0.80      0.86      0.83        43
        390       0.99      0.99      0.99       158
        391       0.63      0.56      0.60        89
        392       0.84      0.92      0.88        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.72      0.79      0.75        80
        396       0.62      0.74      0.68        58
        397       0.96      0.93      0.94       121
        398       0.90      0.94      0.92        80
        399       0.95      0.93      0.94        44
        400       0.43      0.68      0.53        37
        401       0.85      0.76      0.80       108
        402       0.94      0.88      0.91        33
        403       0.93      0.88      0.90       123
        404       0.87      0.95      0.90        55
        405       0.90      0.71      0.79       102
        406       0.81      0.84      0.83       110
        407       0.61      0.62      0.61        76
        408       0.62      0.93      0.75        30
        409       0.47      0.60      0.53        42
        410       0.77      0.39      0.52        62
        411       0.73      0.70      0.71       135
        412       0.46      0.52      0.49        64
        413       0.97      0.97      0.97        31
        414       0.95      0.92      0.94        66
        415       0.28      0.25      0.26        88
        416       0.85      0.80      0.83       115
        417       0.97      0.95      0.96       124
        418       0.90      0.77      0.83       126
        419       0.78      0.47      0.58        75
        420       0.90      0.92      0.91        48
        421       0.91      1.00      0.95        30
        422       0.50      0.81      0.62        42
        423       0.65      0.67      0.66       109
        424       0.88      1.00      0.94        30
        425       0.77      0.89      0.82        37
        426       0.44      0.65      0.53        31
        427       0.69      0.97      0.81        32
        428       0.53      0.70      0.60        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       1.00      0.98      0.99        42
        432       0.65      0.67      0.66        70
        433       0.37      0.49      0.42        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.54      0.78      0.64        64
        437       0.70      0.88      0.78        32
        438       0.82      0.68      0.74        53
        439       0.68      0.70      0.69        30
        440       0.96      0.96      0.96        85
        441       0.35      0.64      0.45        33
        442       0.78      0.80      0.79       131
        443       0.53      0.60      0.56        85
        444       0.56      0.70      0.62        47
        445       0.82      0.87      0.85        38
        446       0.96      0.84      0.90       122
        447       0.65      0.57      0.61        63
        448       0.91      0.91      0.91        57
        449       0.89      0.98      0.93        57
        450       0.55      0.71      0.62        56
        451       0.64      0.69      0.66        51
        452       0.45      0.50      0.48        40
        453       0.95      0.91      0.93       137
        454       0.97      0.92      0.95        39
        455       0.73      0.87      0.80        63
        456       0.90      0.93      0.91        84
        457       0.91      0.78      0.84       136
        458       0.55      0.96      0.70        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.58        46
        462       0.51      0.47      0.49        68
        463       0.78      0.81      0.80       120
        464       0.88      0.83      0.85        93
        465       0.63      0.82      0.71        62
        466       0.47      0.77      0.59        31
        467       0.75      0.68      0.72        72
        468       0.91      0.98      0.94        41
        469       0.97      0.88      0.93        78
        470       0.97      0.92      0.94        36
        471       0.40      0.05      0.08        44
        472       0.64      0.72      0.68       112
        473       0.92      0.94      0.93        35
        474       0.50      0.49      0.49        78
        475       0.58      0.83      0.68        30
        476       0.91      0.86      0.88        78
        477       0.87      0.78      0.83       106
        478       0.75      0.47      0.58        32
        479       0.77      0.90      0.83        63
        480       0.94      0.98      0.96        91
        481       0.79      0.84      0.82        45
        482       0.88      0.94      0.91        32
        483       0.99      0.92      0.95       166
        484       0.94      1.00      0.97        34
        485       0.62      0.91      0.74        35
        486       0.34      0.33      0.33        73
        487       0.64      0.66      0.65        59
        488       0.46      0.84      0.60        31
        489       0.79      0.94      0.86        48
        490       0.63      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.78      0.81       114
        493       0.97      0.87      0.92       149
        494       0.67      0.67      0.67        51
        495       0.89      0.88      0.88        80
        496       0.93      1.00      0.96        38
        497       0.48      0.51      0.49        88

avg / total       0.77      0.76      0.76     35492

Iteration number, batch number :  5 35
Iteration number, batch number :  5 34
Iteration number, batch number :  5 33
Iteration number, batch number :  5 32
Iteration number, batch number :  5 31
Iteration number, batch number :  5 30
Iteration number, batch number :  5 29
Iteration number, batch number :  5 28
Iteration number, batch number :  5 27
Iteration number, batch number :  5 26
Iteration number, batch number :  5 25
Iteration number, batch number :  5 24
Iteration number, batch number :  5 23
Iteration number, batch number :  5 22
Iteration number, batch number :  5 21
Iteration number, batch number :  5 20
Iteration number, batch number :  5 19
Iteration number, batch number :  5 18
Iteration number, batch number :  5 17
Iteration number, batch number :  5 16
Iteration number, batch number :  5 15
Iteration number, batch number :  5 14
Iteration number, batch number :  5 13
Iteration number, batch number :  5 12
Iteration number, batch number :  5 11
Iteration number, batch number :  5 10
Iteration number, batch number :  5 9
Iteration number, batch number :  5 8
Iteration number, batch number :  5 7
Iteration number, batch number :  5 6
Iteration number, batch number :  5 5
Iteration number, batch number :  5 4
Iteration number, batch number :  5 3
Iteration number, batch number :  5 2
Iteration number, batch number :  5 1
Iteration number, batch number :  5 0
Accuracy on cv data :  79.2760029094 28338.0 28338
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.75      0.90      0.82        42
          2       0.64      0.92      0.76        37
          3       0.96      1.00      0.98        53
          4       0.97      0.85      0.91       131
          5       0.87      0.92      0.89        37
          6       0.45      0.65      0.53        51
          7       0.68      0.55      0.61        47
          8       0.62      0.55      0.58        77
          9       0.98      1.00      0.99        51
         10       0.96      0.93      0.95        46
         11       0.78      0.88      0.82        32
         12       0.86      0.94      0.90        79
         13       0.77      0.89      0.83        85
         14       0.93      0.97      0.95       143
         15       0.47      0.36      0.41       121
         16       0.85      0.94      0.89        35
         17       0.83      0.74      0.78       125
         18       0.83      0.81      0.82        70
         19       0.66      0.76      0.71        92
         20       0.69      0.74      0.72        39
         21       0.96      0.96      0.96        53
         22       0.72      0.72      0.72        46
         23       0.60      0.91      0.72        33
         24       0.75      0.90      0.82        40
         25       0.98      0.99      0.98        86
         26       0.96      0.90      0.93       132
         27       0.80      0.86      0.83        90
         28       0.54      0.84      0.66        31
         29       0.89      0.85      0.87        96
         30       0.86      0.88      0.87        74
         31       0.97      0.97      0.97        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.61      0.78      0.68        32
         35       0.88      0.90      0.89        77
         36       0.91      0.95      0.93        66
         37       0.75      0.77      0.76       115
         38       0.14      0.04      0.06       121
         39       0.89      0.91      0.90        35
         40       0.36      0.38      0.37        76
         41       0.89      0.83      0.86        93
         42       0.65      0.83      0.73        59
         43       0.92      0.95      0.94        63
         44       0.68      0.84      0.75        31
         45       0.86      0.93      0.89        40
         46       0.86      0.61      0.71       161
         47       0.60      0.75      0.67        48
         48       0.91      0.84      0.87        37
         49       0.74      0.89      0.81        38
         50       0.81      0.73      0.77       124
         51       0.53      0.41      0.46       101
         52       0.96      0.96      0.96        51
         53       0.97      0.99      0.98        77
         54       0.94      0.94      0.94        71
         55       1.00      0.96      0.98        95
         56       0.57      0.53      0.55        66
         57       0.80      0.75      0.77        80
         58       0.98      0.99      0.98        99
         59       0.97      0.90      0.93        93
         60       0.97      0.94      0.96        34
         61       0.85      1.00      0.92        41
         62       0.74      0.84      0.79        31
         63       0.99      0.92      0.95        74
         64       0.82      0.97      0.89        38
         65       0.91      0.88      0.89        96
         66       0.81      0.87      0.84        95
         67       0.86      0.90      0.88       132
         68       0.55      0.56      0.56        73
         69       0.41      0.40      0.41        30
         70       0.81      0.88      0.84        43
         71       0.61      0.64      0.62        44
         72       0.46      0.62      0.53        42
         73       0.94      0.84      0.89        37
         74       0.71      0.86      0.78        99
         75       0.89      0.87      0.88       164
         76       0.93      0.95      0.94        65
         77       0.94      0.96      0.95        71
         78       0.93      0.76      0.84        34
         79       0.70      0.85      0.77        33
         80       0.75      0.85      0.80        72
         81       0.92      0.98      0.95       132
         82       0.86      0.63      0.73       175
         83       0.78      0.73      0.75        48
         84       0.43      0.44      0.43        45
         85       1.00      0.96      0.98       110
         86       0.65      0.54      0.59        98
         87       0.80      0.73      0.77        83
         88       0.83      0.88      0.85        43
         89       0.84      0.91      0.88        35
         90       0.79      0.82      0.81        38
         91       0.66      0.53      0.59        51
         92       0.32      0.49      0.39        41
         93       0.88      0.92      0.90        48
         94       0.96      0.92      0.94        71
         95       0.83      0.82      0.83       117
         96       0.65      0.89      0.76        38
         97       0.67      0.84      0.74        31
         98       0.95      0.88      0.91        99
         99       0.76      0.75      0.76        69
        100       0.90      0.90      0.90        99
        101       0.65      0.60      0.63        85
        102       0.53      0.62      0.57        56
        103       0.94      0.94      0.94        36
        104       0.70      0.76      0.73        41
        105       0.72      0.65      0.68        65
        106       0.72      0.81      0.76        32
        107       0.93      0.74      0.83        35
        108       0.69      0.72      0.71        82
        109       0.93      0.81      0.86        78
        110       0.98      0.95      0.97       121
        111       0.67      0.76      0.71        54
        112       0.96      0.97      0.96        89
        113       0.74      0.94      0.83        31
        114       0.85      0.87      0.86       127
        115       0.96      0.94      0.95        78
        116       0.61      0.47      0.53       108
        117       0.86      0.88      0.87       110
        118       0.51      0.41      0.46        88
        119       0.89      0.77      0.82       107
        120       0.57      0.61      0.59        70
        121       0.77      0.95      0.85        39
        122       0.95      0.85      0.90       124
        123       0.88      0.76      0.82        59
        124       0.93      0.96      0.94        70
        125       0.78      0.92      0.85        39
        126       0.80      0.72      0.76       114
        127       1.00      1.00      1.00        56
        128       0.68      0.86      0.76        83
        129       0.97      0.91      0.94       125
        130       0.77      0.83      0.80        48
        131       0.84      0.88      0.86       112
        132       1.00      0.97      0.98        97
        133       0.99      0.74      0.85       164
        134       0.41      0.56      0.47        32
        135       0.82      0.88      0.85        81
        136       0.74      0.73      0.73       104
        137       0.70      0.69      0.70       134
        138       0.99      0.93      0.96        88
        139       0.33      0.13      0.19        38
        140       0.98      0.91      0.94        44
        141       0.88      0.84      0.86        62
        142       1.00      0.92      0.96       104
        143       0.62      0.71      0.66        51
        144       0.53      0.60      0.56        73
        145       0.69      0.86      0.77        50
        146       0.85      0.80      0.83        86
        147       0.88      0.88      0.88        93
        148       0.94      0.78      0.85        58
        149       0.92      0.90      0.91       110
        150       0.78      0.79      0.79        77
        151       0.65      0.63      0.64        75
        152       0.95      0.96      0.95        54
        153       0.48      0.94      0.63        32
        154       0.79      0.77      0.78        74
        155       0.89      0.72      0.80       189
        156       0.61      0.47      0.53        92
        157       0.88      0.94      0.91        31
        158       0.76      0.71      0.73       101
        159       0.54      0.57      0.56       109
        160       0.18      0.26      0.21        39
        161       0.68      0.74      0.71        57
        162       0.89      0.97      0.93        33
        163       0.41      0.39      0.40        70
        164       0.96      0.86      0.91       160
        165       0.34      0.38      0.36        40
        166       0.89      0.93      0.91        72
        167       0.75      0.84      0.79        64
        168       0.88      0.75      0.81        76
        169       0.75      0.82      0.79        40
        170       0.99      0.95      0.97       141
        171       0.17      0.06      0.09        62
        172       0.86      0.88      0.87        89
        173       0.65      0.72      0.68        39
        174       0.74      0.92      0.82        38
        175       0.80      0.96      0.87        45
        176       0.41      0.60      0.49        43
        177       0.67      0.88      0.76        32
        178       0.94      0.91      0.93        55
        179       0.46      0.56      0.51        32
        180       0.66      0.63      0.65        82
        181       0.61      0.59      0.60        32
        182       0.74      0.68      0.71        87
        183       0.68      0.86      0.76        42
        184       0.88      0.88      0.88        58
        185       0.74      0.87      0.80        30
        186       0.96      0.95      0.95        56
        187       0.67      0.97      0.79        31
        188       0.97      1.00      0.99        33
        189       0.77      0.91      0.83        47
        190       0.95      0.93      0.94       108
        191       0.98      1.00      0.99        40
        192       0.97      0.94      0.95       119
        193       0.97      1.00      0.99        33
        194       0.77      0.81      0.79       152
        195       0.94      0.83      0.88        71
        196       0.92      0.92      0.92        39
        197       0.94      0.94      0.94       124
        198       0.89      0.77      0.83        31
        199       0.47      0.55      0.51        38
        200       0.94      0.97      0.96        34
        201       0.31      0.35      0.33        31
        202       0.82      0.85      0.83        53
        203       0.86      0.91      0.89        34
        204       0.58      0.77      0.66        39
        205       0.62      0.71      0.66        75
        206       0.98      0.89      0.93       101
        207       0.33      0.36      0.34        45
        208       0.81      0.71      0.76       108
        209       0.90      0.94      0.92        80
        210       0.78      0.85      0.81        33
        211       0.86      0.95      0.90        60
        212       0.74      0.68      0.71        93
        213       0.80      0.76      0.78        87
        214       0.21      0.17      0.18        36
        215       0.47      0.47      0.47        92
        216       0.76      0.83      0.79        75
        217       0.53      0.58      0.55        71
        218       0.51      0.61      0.55        56
        219       0.87      0.77      0.81        60
        220       0.81      0.86      0.83        50
        221       0.56      0.65      0.60        43
        222       1.00      1.00      1.00        46
        223       0.74      0.71      0.73        97
        224       1.00      0.84      0.91        85
        225       0.44      0.73      0.55        33
        226       0.91      1.00      0.96        43
        227       0.98      0.94      0.96        54
        228       0.53      0.80      0.64        30
        229       0.35      0.69      0.47        32
        230       0.58      0.59      0.59        37
        231       0.94      0.83      0.88        59
        232       0.92      0.86      0.89        98
        233       0.94      0.93      0.94        70
        234       0.96      0.97      0.96        69
        235       0.90      0.86      0.88       104
        236       1.00      0.96      0.98        99
        237       0.93      0.86      0.89       123
        238       0.90      0.90      0.90        93
        239       1.00      0.96      0.98        48
        240       0.58      0.63      0.61        82
        241       0.76      0.85      0.80        33
        242       0.82      0.85      0.84        95
        243       0.86      0.89      0.88        63
        244       0.42      0.54      0.47        41
        245       0.78      0.70      0.74       104
        246       0.84      0.69      0.76        77
        247       0.95      1.00      0.98        62
        248       0.97      0.95      0.96        78
        249       0.87      0.75      0.81        61
        250       0.92      0.84      0.88       121
        251       0.63      0.55      0.59        31
        252       0.74      0.93      0.82        42
        253       0.59      0.67      0.62        45
        254       0.22      0.29      0.25        48
        255       0.98      1.00      0.99        42
        256       1.00      0.98      0.99        98
        257       0.88      0.94      0.91        32
        258       0.47      0.53      0.50        55
        259       0.92      0.96      0.94       128
        260       0.64      0.49      0.55       121
        261       0.58      0.85      0.69        39
        262       0.98      1.00      0.99        62
        263       0.49      0.67      0.57        64
        264       0.93      0.93      0.93        45
        265       0.87      0.95      0.91        56
        266       0.90      0.94      0.92        68
        267       0.87      0.87      0.87        67
        268       0.83      0.81      0.82       148
        269       0.61      0.81      0.69        31
        270       0.66      0.42      0.51       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.96      1.00      0.98        44
        274       0.79      0.77      0.78       168
        275       0.30      0.63      0.41        38
        276       0.25      0.51      0.33        35
        277       0.87      0.95      0.91        58
        278       0.68      0.76      0.71        33
        279       0.52      0.57      0.55        63
        280       0.95      0.88      0.91        67
        281       0.65      0.82      0.73        45
        282       0.78      0.94      0.85        49
        283       0.82      1.00      0.90        32
        284       0.72      0.61      0.66        96
        285       0.76      0.80      0.78       118
        286       0.86      0.86      0.86        66
        287       0.74      0.67      0.71       107
        288       0.92      0.94      0.93        35
        289       0.93      0.88      0.90        88
        290       0.82      0.83      0.83        66
        291       0.99      0.88      0.93        96
        292       0.99      0.81      0.89       135
        293       0.73      0.84      0.78        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.64      0.59      0.61        82
        297       0.91      0.84      0.87       107
        298       0.74      0.78      0.76        41
        299       0.86      0.81      0.83        31
        300       0.77      0.88      0.82        42
        301       0.73      0.83      0.78        53
        302       0.95      0.93      0.94        90
        303       0.57      0.77      0.66        35
        304       0.91      0.98      0.94        42
        305       0.94      0.89      0.91       102
        306       1.00      0.95      0.97        39
        307       0.88      0.76      0.81       120
        308       0.96      0.99      0.97       117
        309       0.91      0.87      0.89        98
        310       0.96      0.99      0.98       110
        311       0.43      0.44      0.44        36
        312       0.96      1.00      0.98        99
        313       0.78      0.94      0.85       116
        314       0.59      0.74      0.66        39
        315       0.62      0.70      0.66        30
        316       0.93      0.83      0.88       156
        317       0.78      0.84      0.81        75
        318       0.66      0.72      0.69        72
        319       0.98      0.90      0.94       154
        320       0.95      0.85      0.90        86
        321       0.27      0.17      0.21        59
        322       0.74      0.80      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.88      0.91        56
        325       0.57      0.57      0.57       112
        326       0.92      0.90      0.91       114
        327       0.86      0.85      0.85        93
        328       0.75      0.72      0.73        85
        329       0.30      0.14      0.19        76
        330       0.63      0.67      0.65        66
        331       0.86      0.90      0.88       128
        332       0.88      0.91      0.90        33
        333       0.50      0.55      0.53        56
        334       0.62      0.44      0.51        41
        335       0.86      0.65      0.74       103
        336       0.76      0.85      0.80        46
        337       0.96      0.93      0.94        54
        338       0.77      0.92      0.84        52
        339       0.71      0.68      0.70        57
        340       0.84      0.76      0.80       128
        341       0.91      0.93      0.92        42
        342       0.79      0.65      0.72       118
        343       0.54      0.58      0.56        45
        344       0.57      0.71      0.64        38
        345       0.99      0.97      0.98       153
        346       0.60      0.78      0.68        65
        347       0.91      0.83      0.87        64
        348       0.75      0.56      0.64        99
        349       1.00      0.99      1.00       105
        350       0.78      0.80      0.79        86
        351       0.94      0.96      0.95       125
        352       0.41      0.38      0.39        64
        353       0.41      0.56      0.47        50
        354       0.41      0.68      0.51        41
        355       0.94      0.85      0.89       108
        356       0.80      0.75      0.77        48
        357       0.70      0.65      0.67        98
        358       0.64      0.63      0.64        43
        359       0.72      0.74      0.73       120
        360       1.00      0.99      1.00       103
        361       0.42      0.43      0.43       120
        362       0.98      0.89      0.93       158
        363       0.91      0.98      0.94        51
        364       0.56      0.76      0.65        42
        365       0.89      0.86      0.88       122
        366       0.72      0.58      0.64       110
        367       1.00      0.94      0.97        67
        368       0.85      0.86      0.86       114
        369       0.66      0.82      0.73        62
        370       0.91      0.93      0.92       130
        371       0.66      0.69      0.68        62
        372       0.81      0.83      0.82       110
        373       0.89      0.86      0.87       145
        374       0.45      0.72      0.55        32
        375       0.47      0.37      0.41       100
        376       0.95      0.95      0.95        39
        377       0.68      0.74      0.71        38
        378       0.38      0.58      0.46        31
        379       0.75      0.81      0.78        54
        380       0.72      0.89      0.80        44
        381       0.33      0.38      0.35        48
        382       0.70      0.84      0.76        31
        383       0.71      0.79      0.75        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.74      0.80      0.77        61
        387       0.59      0.75      0.66        32
        388       0.69      0.67      0.68       115
        389       0.76      0.86      0.80        43
        390       0.99      0.97      0.98       158
        391       0.64      0.72      0.68        89
        392       0.84      0.96      0.90        50
        393       0.93      0.93      0.93        69
        394       0.83      0.87      0.85        55
        395       0.80      0.85      0.83        81
        396       0.75      0.75      0.75        59
        397       0.94      0.97      0.95       121
        398       0.88      0.94      0.91        80
        399       0.81      0.87      0.84        45
        400       0.54      0.50      0.52        38
        401       0.76      0.84      0.80       108
        402       0.91      0.91      0.91        34
        403       0.90      0.84      0.87       123
        404       0.84      0.89      0.87        55
        405       0.88      0.83      0.86       103
        406       0.78      0.86      0.82       111
        407       0.72      0.76      0.74        76
        408       0.62      0.87      0.72        30
        409       0.56      0.60      0.57        42
        410       0.78      0.44      0.57        63
        411       0.82      0.76      0.79       136
        412       0.59      0.52      0.55        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.38      0.38        88
        416       0.90      0.70      0.79       116
        417       0.94      0.94      0.94       124
        418       0.91      0.72      0.80       127
        419       0.83      0.89      0.86        76
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.63      0.81      0.71        42
        423       0.75      0.70      0.73       110
        424       0.69      0.97      0.81        30
        425       0.75      0.89      0.81        37
        426       0.69      0.78      0.74        32
        427       0.75      0.94      0.83        32
        428       0.48      0.67      0.56        55
        429       0.96      0.98      0.97        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.77      0.65      0.70        71
        433       0.49      0.51      0.50        55
        434       0.88      0.94      0.91        32
        435       0.92      0.82      0.86        66
        436       0.90      0.86      0.88        65
        437       0.70      0.94      0.80        32
        438       0.76      0.65      0.70        54
        439       0.78      0.70      0.74        30
        440       0.98      0.99      0.98        86
        441       0.38      0.61      0.47        33
        442       0.80      0.77      0.78       131
        443       0.52      0.63      0.57        86
        444       0.50      0.46      0.48        48
        445       0.75      0.95      0.84        38
        446       0.97      0.99      0.98       123
        447       0.61      0.75      0.67        63
        448       0.86      0.76      0.81        58
        449       0.87      0.95      0.91        57
        450       0.58      0.71      0.64        56
        451       0.70      0.78      0.74        51
        452       0.53      0.73      0.61        41
        453       0.96      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.79      0.87      0.83        63
        456       0.93      0.81      0.87        85
        457       0.91      0.78      0.84       137
        458       0.75      0.81      0.78        47
        459       0.60      0.38      0.46        87
        460       0.99      0.99      0.99        95
        461       0.52      0.76      0.62        46
        462       0.56      0.51      0.53        69
        463       0.89      0.72      0.79       120
        464       0.86      0.83      0.84        93
        465       0.72      0.89      0.79        63
        466       0.40      0.56      0.47        32
        467       0.77      0.70      0.73        73
        468       0.93      0.98      0.95        42
        469       0.91      0.87      0.89        78
        470       1.00      0.97      0.99        36
        471       0.81      0.66      0.73        44
        472       0.67      0.65      0.66       112
        473       0.87      0.94      0.91        36
        474       0.53      0.44      0.48        79
        475       0.45      0.74      0.56        31
        476       0.97      0.88      0.93        78
        477       0.93      0.82      0.87       107
        478       0.95      0.58      0.72        33
        479       0.88      0.94      0.91        63
        480       0.91      0.95      0.93        91
        481       0.71      0.76      0.74        46
        482       0.91      0.88      0.89        33
        483       0.99      0.86      0.92       167
        484       1.00      0.97      0.99        35
        485       0.82      0.75      0.78        36
        486       0.28      0.30      0.29        74
        487       0.72      0.77      0.74        60
        488       0.55      0.68      0.61        31
        489       0.92      0.92      0.92        49
        490       0.82      0.63      0.71       121
        491       0.82      0.85      0.84        39
        492       0.74      0.74      0.74       114
        493       0.95      0.85      0.89       149
        494       0.64      0.73      0.68        51
        495       0.91      0.93      0.92        80
        496       0.81      0.97      0.88        39
        497       0.59      0.47      0.52        88

avg / total       0.80      0.79      0.79     35746

Iteration number, batch number :  6 0
Training data accuracy :  0.839357429719
Training data loss     :  0.00524387744752
Iteration number, batch number :  6 1
Training data accuracy :  0.824297188755
Training data loss     :  0.00533128571794
Iteration number, batch number :  6 2
Training data accuracy :  0.827309236948
Training data loss     :  0.00529665321953
Iteration number, batch number :  6 3
Training data accuracy :  0.836345381526
Training data loss     :  0.00530049538482
Iteration number, batch number :  6 4
Training data accuracy :  0.841365461847
Training data loss     :  0.00527129139293
Iteration number, batch number :  6 5
Training data accuracy :  0.857429718876
Training data loss     :  0.00507169170036
Iteration number, batch number :  6 6
Training data accuracy :  0.839357429719
Training data loss     :  0.00546018438822
Iteration number, batch number :  6 7
Training data accuracy :  0.855421686747
Training data loss     :  0.00465508825878
Iteration number, batch number :  6 8
Training data accuracy :  0.83734939759
Training data loss     :  0.00508661297127
Iteration number, batch number :  6 9
Training data accuracy :  0.84437751004
Training data loss     :  0.00515444493599
Iteration number, batch number :  6 10
Training data accuracy :  0.867469879518
Training data loss     :  0.00458911452986
Iteration number, batch number :  6 11
Training data accuracy :  0.84437751004
Training data loss     :  0.00499401377718
Iteration number, batch number :  6 12
Training data accuracy :  0.861445783133
Training data loss     :  0.00492397868066
Iteration number, batch number :  6 13
Training data accuracy :  0.836345381526
Training data loss     :  0.00539873466239
Iteration number, batch number :  6 14
Training data accuracy :  0.854417670683
Training data loss     :  0.00474136262805
Iteration number, batch number :  6 15
Training data accuracy :  0.852409638554
Training data loss     :  0.00465668311693
Iteration number, batch number :  6 16
Training data accuracy :  0.873493975904
Training data loss     :  0.00436868917883
Iteration number, batch number :  6 17
Training data accuracy :  0.852409638554
Training data loss     :  0.0046511808405
Iteration number, batch number :  6 18
Training data accuracy :  0.848393574297
Training data loss     :  0.00504543508674
Iteration number, batch number :  6 19
Training data accuracy :  0.863453815261
Training data loss     :  0.00486380076293
Iteration number, batch number :  6 20
Training data accuracy :  0.869477911647
Training data loss     :  0.00458737190722
Iteration number, batch number :  6 21
Training data accuracy :  0.841365461847
Training data loss     :  0.00468068982304
Iteration number, batch number :  6 22
Training data accuracy :  0.859437751004
Training data loss     :  0.00443267568004
Iteration number, batch number :  6 23
Training data accuracy :  0.853413654618
Training data loss     :  0.00440937734542
Iteration number, batch number :  6 24
Training data accuracy :  0.848393574297
Training data loss     :  0.00488096721042
Iteration number, batch number :  6 25
Training data accuracy :  0.848393574297
Training data loss     :  0.00495469433495
Iteration number, batch number :  6 26
Training data accuracy :  0.868473895582
Training data loss     :  0.00466773718744
Iteration number, batch number :  6 27
Training data accuracy :  0.852409638554
Training data loss     :  0.00473720619116
Iteration number, batch number :  6 28
Training data accuracy :  0.873493975904
Training data loss     :  0.00431991847903
Iteration number, batch number :  6 29
Training data accuracy :  0.853413654618
Training data loss     :  0.0047996809061
Iteration number, batch number :  6 30
Training data accuracy :  0.849397590361
Training data loss     :  0.00463815994433
Iteration number, batch number :  6 31
Training data accuracy :  0.845381526104
Training data loss     :  0.0048956933903
Iteration number, batch number :  6 32
Training data accuracy :  0.853413654618
Training data loss     :  0.00476791542442
Iteration number, batch number :  6 33
Training data accuracy :  0.885542168675
Training data loss     :  0.00438384440654
Iteration number, batch number :  6 34
Training data accuracy :  0.880522088353
Training data loss     :  0.00424718892995
Iteration number, batch number :  6 35
Training data accuracy :  0.874497991968
Training data loss     :  0.00410671471026
Iteration number, batch number :  6 36
Training data accuracy :  0.85843373494
Training data loss     :  0.00434501732818
Iteration number, batch number :  6 37
Training data accuracy :  0.878514056225
Training data loss     :  0.00441835090816
Iteration number, batch number :  6 38
Training data accuracy :  0.870481927711
Training data loss     :  0.00445026093151
Iteration number, batch number :  6 39
Training data accuracy :  0.892570281124
Training data loss     :  0.0039150669445
Iteration number, batch number :  6 40
Training data accuracy :  0.856425702811
Training data loss     :  0.00480287663292
Iteration number, batch number :  6 41
Training data accuracy :  0.862449799197
Training data loss     :  0.00439478328242
Iteration number, batch number :  6 42
Training data accuracy :  0.876506024096
Training data loss     :  0.0043624864059
Iteration number, batch number :  6 43
Training data accuracy :  0.874497991968
Training data loss     :  0.00437398550277
Iteration number, batch number :  6 44
Training data accuracy :  0.870481927711
Training data loss     :  0.0039905080497
Iteration number, batch number :  6 45
Training data accuracy :  0.875502008032
Training data loss     :  0.0042197510305
Iteration number, batch number :  6 46
Training data accuracy :  0.868473895582
Training data loss     :  0.00417447611331
Iteration number, batch number :  6 47
Training data accuracy :  0.877510040161
Training data loss     :  0.00451271305945
Iteration number, batch number :  6 48
Training data accuracy :  0.855421686747
Training data loss     :  0.0046714926008
Iteration number, batch number :  6 49
Training data accuracy :  0.856425702811
Training data loss     :  0.00436460513639
Iteration number, batch number :  6 50
Training data accuracy :  0.869477911647
Training data loss     :  0.00440368457186
Iteration number, batch number :  6 51
Training data accuracy :  0.843373493976
Training data loss     :  0.00476023751148
Iteration number, batch number :  6 52
Training data accuracy :  0.85140562249
Training data loss     :  0.00525435699682
Iteration number, batch number :  6 53
Training data accuracy :  0.848393574297
Training data loss     :  0.00496077335332
Iteration number, batch number :  6 54
Training data accuracy :  0.852409638554
Training data loss     :  0.00493965641923
Iteration number, batch number :  6 55
Training data accuracy :  0.845381526104
Training data loss     :  0.00478687095594
Iteration number, batch number :  6 56
Training data accuracy :  0.857429718876
Training data loss     :  0.00501670416584
Iteration number, batch number :  6 57
Training data accuracy :  0.843373493976
Training data loss     :  0.00512463181593
Iteration number, batch number :  6 58
Training data accuracy :  0.832329317269
Training data loss     :  0.00511018694496
Iteration number, batch number :  6 59
Training data accuracy :  0.846385542169
Training data loss     :  0.0053551381616
Iteration number, batch number :  6 60
Training data accuracy :  0.847389558233
Training data loss     :  0.00504097561103
Iteration number, batch number :  6 61
Training data accuracy :  0.839357429719
Training data loss     :  0.00522475074349
Iteration number, batch number :  6 62
Training data accuracy :  0.842369477912
Training data loss     :  0.00515751965211
Iteration number, batch number :  6 63
Training data accuracy :  0.839357429719
Training data loss     :  0.00528316117327
Iteration number, batch number :  6 64
Training data accuracy :  0.847389558233
Training data loss     :  0.00553766539557
Iteration number, batch number :  6 65
Training data accuracy :  0.83734939759
Training data loss     :  0.00571368583104
Iteration number, batch number :  6 66
Training data accuracy :  0.842369477912
Training data loss     :  0.00504450770768
Iteration number, batch number :  6 67
Training data accuracy :  0.856425702811
Training data loss     :  0.00520509612418
Iteration number, batch number :  6 68
Training data accuracy :  0.842369477912
Training data loss     :  0.00543565199183
Iteration number, batch number :  6 69
Training data accuracy :  0.83734939759
Training data loss     :  0.00514881257124
Iteration number, batch number :  6 35
Iteration number, batch number :  6 34
Iteration number, batch number :  6 33
Iteration number, batch number :  6 32
Iteration number, batch number :  6 31
Iteration number, batch number :  6 30
Iteration number, batch number :  6 29
Iteration number, batch number :  6 28
Iteration number, batch number :  6 27
Iteration number, batch number :  6 26
Iteration number, batch number :  6 25
Iteration number, batch number :  6 24
Iteration number, batch number :  6 23
Iteration number, batch number :  6 22
Iteration number, batch number :  6 21
Iteration number, batch number :  6 20
Iteration number, batch number :  6 19
Iteration number, batch number :  6 18
Iteration number, batch number :  6 17
Iteration number, batch number :  6 16
Iteration number, batch number :  6 15
Iteration number, batch number :  6 14
Iteration number, batch number :  6 13
Iteration number, batch number :  6 12
Iteration number, batch number :  6 11
Iteration number, batch number :  6 10
Iteration number, batch number :  6 9
Iteration number, batch number :  6 8
Iteration number, batch number :  6 7
Iteration number, batch number :  6 6
Iteration number, batch number :  6 5
Iteration number, batch number :  6 4
Iteration number, batch number :  6 3
Iteration number, batch number :  6 2
Iteration number, batch number :  6 1
Iteration number, batch number :  6 0
Accuracy on test data :  76.4200383185 27123.0 27123
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.77      0.91      0.84        45
          1       0.75      0.95      0.84        41
          2       0.64      0.81      0.71        37
          3       0.82      0.96      0.88        52
          4       0.95      0.55      0.69       130
          5       0.92      0.97      0.95        36
          6       0.49      0.71      0.58        51
          7       0.71      0.62      0.66        47
          8       0.38      0.38      0.38        76
          9       0.96      1.00      0.98        51
         10       0.95      0.93      0.94        45
         11       0.35      0.45      0.39        31
         12       0.93      0.85      0.89        79
         13       0.71      0.82      0.77        85
         14       0.92      0.85      0.89       143
         15       0.51      0.42      0.46       120
         16       0.91      0.94      0.93        34
         17       0.66      0.31      0.43       124
         18       0.78      0.81      0.80        70
         19       0.53      0.55      0.54        91
         20       0.40      0.62      0.48        39
         21       0.90      0.36      0.51        53
         22       0.53      0.51      0.52        45
         23       0.48      0.73      0.58        33
         24       0.79      0.78      0.78        40
         25       0.98      0.98      0.98        85
         26       0.92      0.73      0.81       132
         27       0.92      0.88      0.90        90
         28       0.37      0.53      0.44        30
         29       0.86      0.84      0.85        96
         30       0.88      0.86      0.87        74
         31       0.97      0.89      0.93        79
         32       1.00      0.99      0.99       269
         33       0.97      0.94      0.95        31
         34       0.41      0.69      0.51        32
         35       0.87      0.95      0.91        77
         36       0.87      0.91      0.89        66
         37       0.76      0.84      0.80       114
         38       0.05      0.01      0.01       121
         39       0.80      0.82      0.81        34
         40       0.34      0.45      0.39        76
         41       0.91      0.64      0.75        92
         42       0.65      0.72      0.68        58
         43       0.95      0.87      0.91        63
         44       0.56      0.80      0.66        30
         45       0.83      0.95      0.88        40
         46       0.00      0.00      0.00       161
         47       0.57      0.77      0.65        47
         48       0.92      0.94      0.93        36
         49       0.83      0.95      0.89        37
         50       0.79      0.65      0.71       123
         51       0.61      0.47      0.53       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.94      0.90      0.92        70
         55       0.98      0.94      0.96        94
         56       0.59      0.64      0.61        66
         57       0.69      0.67      0.68        79
         58       0.97      0.92      0.94        98
         59       0.95      0.90      0.93        92
         60       1.00      0.88      0.94        33
         61       0.88      0.93      0.90        41
         62       0.40      0.07      0.11        30
         63       0.84      0.88      0.86        74
         64       0.86      1.00      0.93        38
         65       0.94      0.88      0.91        95
         66       0.80      0.70      0.75        94
         67       0.88      0.90      0.89       132
         68       0.38      0.43      0.41        72
         69       0.35      0.50      0.41        30
         70       0.74      0.81      0.77        42
         71       0.66      0.80      0.72        44
         72       0.34      0.67      0.45        42
         73       0.82      0.84      0.83        37
         74       0.64      0.89      0.74        98
         75       0.95      0.87      0.91       163
         76       0.91      0.91      0.91        65
         77       0.97      0.94      0.96        70
         78       0.88      0.85      0.86        33
         79       0.59      0.79      0.68        33
         80       0.76      0.66      0.71        71
         81       0.96      0.97      0.97       131
         82       0.83      0.69      0.75       174
         83       0.71      0.74      0.73        47
         84       0.23      0.27      0.25        44
         85       0.96      0.93      0.94       110
         86       0.52      0.45      0.49        97
         87       0.92      0.78      0.84        83
         88       0.77      0.84      0.80        43
         89       0.71      0.91      0.80        35
         90       0.72      0.82      0.77        38
         91       0.69      0.61      0.65        51
         92       0.45      0.72      0.55        40
         93       0.94      0.92      0.93        48
         94       0.89      0.90      0.90        71
         95       0.84      0.79      0.81       117
         96       0.55      0.84      0.67        38
         97       0.71      0.90      0.79        30
         98       0.88      0.85      0.86        98
         99       0.62      0.75      0.68        69
        100       0.92      0.91      0.91        98
        101       0.62      0.65      0.64        85
        102       0.40      0.57      0.47        56
        103       0.97      0.92      0.94        36
        104       0.69      0.60      0.64        40
        105       0.65      0.55      0.60        65
        106       0.63      0.75      0.69        32
        107       1.00      0.53      0.69        34
        108       0.69      0.75      0.72        81
        109       0.61      0.22      0.32        77
        110       0.96      0.91      0.93       121
        111       0.58      0.57      0.57        53
        112       0.91      0.89      0.90        88
        113       0.76      0.97      0.85        30
        114       0.85      0.88      0.86       127
        115       0.92      0.73      0.81        78
        116       0.59      0.52      0.55       108
        117       0.92      0.83      0.87       110
        118       0.51      0.41      0.46        88
        119       0.90      0.80      0.85       106
        120       0.63      0.77      0.69        69
        121       0.78      0.82      0.80        39
        122       0.88      0.77      0.82       124
        123       0.84      0.83      0.84        59
        124       1.00      0.96      0.98        69
        125       0.73      0.87      0.80        38
        126       0.76      0.64      0.69       113
        127       0.98      1.00      0.99        55
        128       0.70      0.85      0.77        82
        129       0.93      0.89      0.91       125
        130       0.57      0.83      0.67        47
        131       0.82      0.36      0.50       112
        132       0.99      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.44      0.59      0.51        32
        135       0.81      0.81      0.81        80
        136       0.73      0.71      0.72       103
        137       0.76      0.81      0.79       134
        138       0.92      0.91      0.91        88
        139       0.16      0.08      0.11        37
        140       0.92      1.00      0.96        44
        141       0.83      0.90      0.87        61
        142       0.98      0.94      0.96       103
        143       0.54      0.69      0.60        51
        144       0.47      0.65      0.55        72
        145       0.72      0.96      0.82        49
        146       0.75      0.75      0.75        85
        147       0.91      0.87      0.89        93
        148       0.86      0.86      0.86        57
        149       0.85      0.86      0.85       109
        150       0.89      0.64      0.75        76
        151       0.68      0.69      0.69        75
        152       0.96      0.91      0.93        54
        153       0.45      0.78      0.57        32
        154       0.90      0.86      0.88        73
        155       0.84      0.59      0.69       189
        156       0.66      0.49      0.56        92
        157       0.91      0.94      0.92        31
        158       0.47      0.15      0.23       100
        159       0.51      0.42      0.46       108
        160       0.24      0.44      0.31        39
        161       0.67      0.74      0.70        57
        162       0.92      1.00      0.96        33
        163       0.38      0.47      0.42        70
        164       0.96      0.81      0.88       159
        165       0.48      0.72      0.58        39
        166       0.92      0.94      0.93        71
        167       0.85      0.80      0.82        64
        168       0.85      0.81      0.83        75
        169       0.53      0.64      0.58        39
        170       0.99      0.94      0.97       140
        171       0.17      0.08      0.11        61
        172       0.78      0.90      0.84        88
        173       0.61      0.69      0.65        39
        174       0.82      0.89      0.86        37
        175       0.77      0.82      0.80        45
        176       0.37      0.50      0.42        42
        177       0.61      0.87      0.72        31
        178       0.93      0.95      0.94        55
        179       0.37      0.58      0.45        31
        180       0.62      0.62      0.62        82
        181       0.57      0.65      0.61        31
        182       0.70      0.67      0.68        87
        183       0.76      0.90      0.82        41
        184       0.87      0.91      0.89        57
        185       0.76      0.87      0.81        30
        186       1.00      0.95      0.97        56
        187       0.74      0.97      0.84        30
        188       0.97      1.00      0.99        33
        189       0.71      0.80      0.76        46
        190       0.91      0.99      0.95       107
        191       0.93      0.95      0.94        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.80      0.74      0.77       152
        195       0.91      0.85      0.88        71
        196       0.95      0.95      0.95        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.39      0.70      0.50        37
        200       0.94      1.00      0.97        34
        201       0.16      0.27      0.20        30
        202       0.73      0.81      0.77        53
        203       0.86      0.97      0.91        33
        204       0.66      0.92      0.77        38
        205       0.51      0.55      0.53        75
        206       0.95      0.92      0.93       101
        207       0.29      0.27      0.28        45
        208       0.79      0.77      0.78       108
        209       0.80      0.85      0.82        79
        210       0.71      0.94      0.81        32
        211       0.92      0.90      0.91        60
        212       0.74      0.71      0.73        93
        213       0.76      0.79      0.78        87
        214       0.23      0.34      0.27        35
        215       0.57      0.56      0.57        91
        216       0.78      0.53      0.63        75
        217       0.53      0.61      0.57        70
        218       0.56      0.69      0.62        55
        219       0.92      0.73      0.81        60
        220       0.82      0.92      0.87        50
        221       0.53      0.69      0.60        42
        222       1.00      0.96      0.98        45
        223       0.80      0.73      0.76        96
        224       0.92      0.85      0.88        85
        225       0.53      0.84      0.65        32
        226       0.95      0.93      0.94        43
        227       0.91      0.98      0.95        54
        228       0.42      0.83      0.56        30
        229       0.27      0.50      0.35        32
        230       0.53      0.62      0.57        37
        231       0.83      0.90      0.86        59
        232       0.96      0.84      0.90        97
        233       0.97      0.94      0.96        69
        234       0.93      0.57      0.70        69
        235       0.85      0.75      0.79       103
        236       0.99      0.98      0.98        98
        237       0.90      0.72      0.80       123
        238       0.86      0.89      0.87        93
        239       1.00      0.96      0.98        48
        240       0.61      0.65      0.63        81
        241       0.86      0.94      0.90        32
        242       0.88      0.75      0.81        95
        243       0.81      0.87      0.84        63
        244       0.42      0.60      0.49        40
        245       0.79      0.77      0.78       103
        246       0.71      0.52      0.60        77
        247       0.94      0.98      0.96        61
        248       0.96      0.90      0.93        78
        249       0.93      0.83      0.88        60
        250       0.94      0.93      0.94       120
        251       0.56      0.73      0.64        30
        252       0.82      0.86      0.84        42
        253       0.55      0.66      0.60        44
        254       0.24      0.32      0.27        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.81      0.94      0.87        31
        258       0.38      0.33      0.35        55
        259       0.96      0.91      0.94       128
        260       0.61      0.42      0.50       121
        261       0.65      0.85      0.73        39
        262       0.97      0.98      0.98        62
        263       0.49      0.70      0.58        63
        264       0.87      0.89      0.88        45
        265       0.70      0.91      0.79        55
        266       0.91      0.94      0.93        67
        267       0.86      0.77      0.82        66
        268       0.80      0.70      0.74       148
        269       0.51      0.90      0.65        30
        270       0.60      0.45      0.51       149
        271       0.81      0.93      0.87        42
        272       0.98      0.95      0.97       119
        273       0.88      1.00      0.93        43
        274       0.80      0.75      0.78       167
        275       0.33      0.46      0.39        37
        276       0.19      0.40      0.26        35
        277       0.93      0.96      0.95        57
        278       0.41      0.28      0.33        32
        279       0.45      0.53      0.49        62
        280       0.00      0.00      0.00        66
        281       0.65      0.87      0.74        45
        282       0.86      0.90      0.88        48
        283       0.79      0.97      0.87        31
        284       0.70      0.66      0.68        96
        285       0.77      0.75      0.76       117
        286       0.97      0.85      0.90        66
        287       0.73      0.71      0.72       107
        288       0.69      0.89      0.78        35
        289       0.95      0.92      0.94        87
        290       0.72      0.73      0.72        66
        291       0.96      0.89      0.92        96
        292       0.97      0.70      0.82       135
        293       0.61      0.91      0.73        43
        294       0.46      0.90      0.61        31
        295       0.78      0.84      0.81        56
        296       0.47      0.53      0.50        81
        297       0.89      0.89      0.89       107
        298       0.67      0.68      0.67        41
        299       0.66      0.87      0.75        31
        300       0.79      0.80      0.80        41
        301       0.69      0.81      0.74        52
        302       0.80      0.91      0.85        90
        303       0.48      0.82      0.61        34
        304       0.93      0.95      0.94        41
        305       0.96      0.87      0.91       101
        306       0.95      0.95      0.95        38
        307       0.90      0.67      0.77       120
        308       0.97      0.96      0.96       117
        309       0.90      0.86      0.88        97
        310       0.95      0.95      0.95       109
        311       0.36      0.46      0.41        35
        312       0.95      0.92      0.93        98
        313       0.67      0.91      0.77       116
        314       0.48      0.72      0.58        39
        315       0.74      0.83      0.78        30
        316       0.95      0.91      0.93       155
        317       0.74      0.85      0.80        75
        318       0.72      0.64      0.68        72
        319       0.98      0.86      0.91       154
        320       0.84      0.82      0.83        85
        321       0.04      0.02      0.02        58
        322       0.76      0.81      0.78       119
        323       1.00      1.00      1.00        43
        324       0.93      0.96      0.95        55
        325       0.60      0.59      0.59       112
        326       0.91      0.87      0.89       114
        327       0.86      0.80      0.83        92
        328       0.70      0.70      0.70        84
        329       0.32      0.16      0.21        75
        330       0.67      0.68      0.68        66
        331       0.93      0.86      0.89       127
        332       0.81      0.91      0.86        33
        333       0.48      0.56      0.52        55
        334       0.50      0.29      0.37        41
        335       0.82      0.78      0.80       103
        336       0.73      0.89      0.80        45
        337       0.91      0.91      0.91        54
        338       0.80      0.88      0.84        51
        339       0.69      0.75      0.72        57
        340       0.77      0.76      0.76       127
        341       0.93      0.95      0.94        41
        342       0.74      0.70      0.72       118
        343       0.60      0.67      0.63        45
        344       0.63      0.89      0.74        37
        345       0.99      0.97      0.98       152
        346       0.58      0.69      0.63        64
        347       0.89      0.90      0.90        63
        348       0.73      0.58      0.65        98
        349       1.00      0.98      0.99       104
        350       0.77      0.85      0.81        86
        351       0.88      0.86      0.87       124
        352       0.45      0.60      0.51        63
        353       0.52      0.52      0.52        50
        354       0.47      0.70      0.57        40
        355       0.95      0.87      0.91       108
        356       0.82      0.89      0.86        47
        357       0.70      0.58      0.63        98
        358       0.65      0.81      0.72        42
        359       0.75      0.83      0.79       120
        360       1.00      0.92      0.96       102
        361       0.49      0.45      0.47       120
        362       0.94      0.93      0.94       157
        363       0.91      0.98      0.94        51
        364       0.45      0.67      0.54        42
        365       0.92      0.89      0.90       122
        366       0.71      0.61      0.65       109
        367       1.00      0.97      0.98        67
        368       0.82      0.85      0.83       113
        369       0.55      0.71      0.62        62
        370       0.98      0.92      0.95       130
        371       0.58      0.72      0.64        61
        372       0.81      0.83      0.82       109
        373       0.82      0.80      0.81       145
        374       0.45      0.71      0.55        31
        375       0.52      0.34      0.41        99
        376       0.95      0.92      0.93        38
        377       0.74      0.76      0.75        37
        378       0.24      0.61      0.35        31
        379       0.79      0.81      0.80        54
        380       0.73      0.91      0.81        44
        381       0.22      0.38      0.28        48
        382       0.68      0.90      0.78        31
        383       0.68      0.74      0.71        89
        384       0.99      1.00      1.00       100
        385       0.98      0.89      0.93        45
        386       0.82      0.83      0.83        60
        387       0.62      0.77      0.69        31
        388       0.67      0.72      0.69       115
        389       0.76      0.86      0.80        43
        390       0.99      0.99      0.99       158
        391       0.66      0.57      0.61        89
        392       0.84      0.92      0.88        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.73      0.80      0.76        80
        396       0.62      0.74      0.68        58
        397       0.96      0.93      0.94       121
        398       0.90      0.94      0.92        80
        399       0.95      0.93      0.94        44
        400       0.44      0.68      0.53        37
        401       0.85      0.76      0.80       108
        402       0.94      0.88      0.91        33
        403       0.93      0.88      0.90       123
        404       0.85      0.95      0.90        55
        405       0.90      0.72      0.80       102
        406       0.82      0.84      0.83       110
        407       0.60      0.62      0.61        76
        408       0.64      0.93      0.76        30
        409       0.48      0.60      0.53        42
        410       0.78      0.40      0.53        62
        411       0.73      0.70      0.71       135
        412       0.49      0.53      0.51        64
        413       0.97      0.97      0.97        31
        414       0.95      0.92      0.94        66
        415       0.28      0.25      0.26        88
        416       0.85      0.81      0.83       115
        417       0.97      0.95      0.96       124
        418       0.89      0.77      0.83       126
        419       0.78      0.47      0.58        75
        420       0.88      0.92      0.90        48
        421       0.91      1.00      0.95        30
        422       0.51      0.81      0.62        42
        423       0.66      0.67      0.67       109
        424       0.88      1.00      0.94        30
        425       0.76      0.92      0.83        37
        426       0.44      0.65      0.53        31
        427       0.70      0.97      0.82        32
        428       0.53      0.70      0.60        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       1.00      0.98      0.99        42
        432       0.66      0.69      0.67        70
        433       0.38      0.49      0.43        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.53      0.78      0.63        64
        437       0.70      0.88      0.78        32
        438       0.82      0.70      0.76        53
        439       0.66      0.70      0.68        30
        440       0.96      0.96      0.96        85
        441       0.35      0.64      0.45        33
        442       0.78      0.81      0.79       131
        443       0.53      0.60      0.56        85
        444       0.55      0.70      0.62        47
        445       0.82      0.87      0.85        38
        446       0.96      0.84      0.90       122
        447       0.62      0.49      0.55        63
        448       0.91      0.91      0.91        57
        449       0.89      1.00      0.94        57
        450       0.56      0.73      0.64        56
        451       0.63      0.71      0.67        51
        452       0.45      0.50      0.48        40
        453       0.95      0.91      0.93       137
        454       0.97      0.92      0.95        39
        455       0.73      0.87      0.80        63
        456       0.91      0.93      0.92        84
        457       0.91      0.78      0.84       136
        458       0.56      0.96      0.71        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.59        46
        462       0.51      0.47      0.49        68
        463       0.78      0.81      0.80       120
        464       0.89      0.83      0.86        93
        465       0.63      0.82      0.71        62
        466       0.46      0.74      0.57        31
        467       0.75      0.69      0.72        72
        468       0.91      0.98      0.94        41
        469       0.97      0.88      0.93        78
        470       0.97      0.92      0.94        36
        471       0.40      0.05      0.08        44
        472       0.63      0.72      0.67       112
        473       0.92      0.94      0.93        35
        474       0.50      0.50      0.50        78
        475       0.60      0.83      0.69        30
        476       0.91      0.86      0.88        78
        477       0.87      0.78      0.83       106
        478       0.75      0.47      0.58        32
        479       0.78      0.90      0.84        63
        480       0.94      0.98      0.96        91
        481       0.79      0.84      0.82        45
        482       0.91      0.94      0.92        32
        483       0.99      0.92      0.95       166
        484       0.94      1.00      0.97        34
        485       0.63      0.91      0.74        35
        486       0.34      0.33      0.33        73
        487       0.63      0.64      0.64        59
        488       0.47      0.84      0.60        31
        489       0.78      0.94      0.85        48
        490       0.63      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.78      0.81       114
        493       0.97      0.89      0.93       149
        494       0.65      0.67      0.66        51
        495       0.89      0.88      0.88        80
        496       0.93      1.00      0.96        38
        497       0.49      0.52      0.51        88

avg / total       0.77      0.76      0.76     35492

Iteration number, batch number :  6 35
Iteration number, batch number :  6 34
Iteration number, batch number :  6 33
Iteration number, batch number :  6 32
Iteration number, batch number :  6 31
Iteration number, batch number :  6 30
Iteration number, batch number :  6 29
Iteration number, batch number :  6 28
Iteration number, batch number :  6 27
Iteration number, batch number :  6 26
Iteration number, batch number :  6 25
Iteration number, batch number :  6 24
Iteration number, batch number :  6 23
Iteration number, batch number :  6 22
Iteration number, batch number :  6 21
Iteration number, batch number :  6 20
Iteration number, batch number :  6 19
Iteration number, batch number :  6 18
Iteration number, batch number :  6 17
Iteration number, batch number :  6 16
Iteration number, batch number :  6 15
Iteration number, batch number :  6 14
Iteration number, batch number :  6 13
Iteration number, batch number :  6 12
Iteration number, batch number :  6 11
Iteration number, batch number :  6 10
Iteration number, batch number :  6 9
Iteration number, batch number :  6 8
Iteration number, batch number :  6 7
Iteration number, batch number :  6 6
Iteration number, batch number :  6 5
Iteration number, batch number :  6 4
Iteration number, batch number :  6 3
Iteration number, batch number :  6 2
Iteration number, batch number :  6 1
Iteration number, batch number :  6 0
Accuracy on cv data :  79.441056342 28397.0 28397
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.75      0.90      0.82        42
          2       0.65      0.92      0.76        37
          3       0.95      1.00      0.97        53
          4       0.97      0.85      0.91       131
          5       0.87      0.92      0.89        37
          6       0.47      0.71      0.57        51
          7       0.69      0.57      0.63        47
          8       0.61      0.55      0.58        77
          9       0.98      1.00      0.99        51
         10       0.96      0.93      0.95        46
         11       0.76      0.88      0.81        32
         12       0.86      0.94      0.90        79
         13       0.77      0.89      0.83        85
         14       0.93      0.97      0.95       143
         15       0.47      0.37      0.41       121
         16       0.85      0.94      0.89        35
         17       0.83      0.74      0.78       125
         18       0.81      0.83      0.82        70
         19       0.65      0.76      0.70        92
         20       0.67      0.74      0.71        39
         21       0.96      0.96      0.96        53
         22       0.72      0.72      0.72        46
         23       0.60      0.91      0.72        33
         24       0.77      0.90      0.83        40
         25       0.99      0.99      0.99        86
         26       0.96      0.90      0.93       132
         27       0.80      0.86      0.83        90
         28       0.54      0.84      0.66        31
         29       0.89      0.88      0.88        96
         30       0.88      0.88      0.88        74
         31       0.97      0.97      0.97        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.61      0.78      0.68        32
         35       0.89      0.92      0.90        77
         36       0.90      0.95      0.93        66
         37       0.75      0.77      0.76       115
         38       0.14      0.04      0.06       121
         39       0.89      0.91      0.90        35
         40       0.36      0.38      0.37        76
         41       0.89      0.83      0.86        93
         42       0.68      0.83      0.75        59
         43       0.92      0.95      0.94        63
         44       0.68      0.84      0.75        31
         45       0.86      0.93      0.89        40
         46       0.86      0.60      0.71       161
         47       0.61      0.77      0.68        48
         48       0.91      0.84      0.87        37
         49       0.74      0.89      0.81        38
         50       0.83      0.73      0.77       124
         51       0.53      0.41      0.46       101
         52       0.96      0.96      0.96        51
         53       0.97      0.99      0.98        77
         54       0.94      0.94      0.94        71
         55       1.00      0.96      0.98        95
         56       0.58      0.53      0.56        66
         57       0.80      0.75      0.77        80
         58       0.98      0.99      0.98        99
         59       0.97      0.90      0.93        93
         60       0.97      0.94      0.96        34
         61       0.85      1.00      0.92        41
         62       0.79      0.84      0.81        31
         63       0.99      0.92      0.95        74
         64       0.82      0.97      0.89        38
         65       0.91      0.88      0.89        96
         66       0.81      0.87      0.84        95
         67       0.87      0.90      0.88       132
         68       0.56      0.55      0.55        73
         69       0.43      0.43      0.43        30
         70       0.79      0.88      0.84        43
         71       0.63      0.66      0.64        44
         72       0.46      0.62      0.53        42
         73       0.91      0.84      0.87        37
         74       0.72      0.86      0.78        99
         75       0.90      0.87      0.88       164
         76       0.93      0.95      0.94        65
         77       0.94      0.96      0.95        71
         78       0.93      0.76      0.84        34
         79       0.70      0.85      0.77        33
         80       0.76      0.86      0.81        72
         81       0.92      0.98      0.95       132
         82       0.87      0.63      0.73       175
         83       0.78      0.73      0.75        48
         84       0.43      0.44      0.43        45
         85       1.00      0.96      0.98       110
         86       0.65      0.55      0.60        98
         87       0.80      0.73      0.77        83
         88       0.83      0.88      0.85        43
         89       0.84      0.91      0.88        35
         90       0.78      0.82      0.79        38
         91       0.64      0.53      0.58        51
         92       0.32      0.49      0.38        41
         93       0.88      0.92      0.90        48
         94       0.96      0.92      0.94        71
         95       0.83      0.82      0.83       117
         96       0.65      0.89      0.76        38
         97       0.66      0.81      0.72        31
         98       0.95      0.87      0.91        99
         99       0.76      0.75      0.76        69
        100       0.90      0.90      0.90        99
        101       0.66      0.61      0.63        85
        102       0.55      0.64      0.59        56
        103       0.94      0.94      0.94        36
        104       0.70      0.76      0.73        41
        105       0.72      0.65      0.68        65
        106       0.68      0.81      0.74        32
        107       0.93      0.74      0.83        35
        108       0.70      0.72      0.71        82
        109       0.94      0.82      0.88        78
        110       0.97      0.95      0.96       121
        111       0.66      0.76      0.71        54
        112       0.96      0.97      0.96        89
        113       0.74      0.94      0.83        31
        114       0.86      0.87      0.87       127
        115       0.96      0.94      0.95        78
        116       0.62      0.48      0.54       108
        117       0.86      0.88      0.87       110
        118       0.51      0.41      0.46        88
        119       0.89      0.77      0.82       107
        120       0.57      0.61      0.59        70
        121       0.77      0.95      0.85        39
        122       0.96      0.85      0.91       124
        123       0.88      0.76      0.82        59
        124       0.93      0.96      0.94        70
        125       0.76      0.90      0.82        39
        126       0.81      0.75      0.78       114
        127       1.00      1.00      1.00        56
        128       0.68      0.87      0.76        83
        129       0.97      0.91      0.94       125
        130       0.78      0.83      0.81        48
        131       0.85      0.88      0.87       112
        132       1.00      0.97      0.98        97
        133       0.99      0.74      0.85       164
        134       0.41      0.56      0.47        32
        135       0.82      0.88      0.85        81
        136       0.74      0.73      0.73       104
        137       0.70      0.69      0.70       134
        138       0.98      0.93      0.95        88
        139       0.33      0.13      0.19        38
        140       0.98      0.91      0.94        44
        141       0.88      0.84      0.86        62
        142       1.00      0.92      0.96       104
        143       0.63      0.73      0.67        51
        144       0.55      0.59      0.57        73
        145       0.69      0.86      0.77        50
        146       0.85      0.81      0.83        86
        147       0.88      0.87      0.88        93
        148       0.94      0.78      0.85        58
        149       0.93      0.90      0.92       110
        150       0.78      0.79      0.79        77
        151       0.66      0.64      0.65        75
        152       0.95      0.96      0.95        54
        153       0.48      0.94      0.64        32
        154       0.78      0.77      0.78        74
        155       0.89      0.72      0.80       189
        156       0.67      0.49      0.57        92
        157       0.88      0.94      0.91        31
        158       0.76      0.72      0.74       101
        159       0.55      0.57      0.56       109
        160       0.18      0.26      0.21        39
        161       0.68      0.74      0.71        57
        162       0.89      0.97      0.93        33
        163       0.40      0.39      0.39        70
        164       0.96      0.86      0.91       160
        165       0.36      0.38      0.37        40
        166       0.89      0.93      0.91        72
        167       0.76      0.84      0.80        64
        168       0.88      0.75      0.81        76
        169       0.75      0.82      0.79        40
        170       0.99      0.95      0.97       141
        171       0.17      0.06      0.09        62
        172       0.86      0.88      0.87        89
        173       0.65      0.72      0.68        39
        174       0.76      0.92      0.83        38
        175       0.80      0.96      0.87        45
        176       0.41      0.60      0.49        43
        177       0.68      0.88      0.77        32
        178       0.94      0.91      0.93        55
        179       0.44      0.56      0.49        32
        180       0.67      0.65      0.66        82
        181       0.56      0.56      0.56        32
        182       0.77      0.68      0.72        87
        183       0.68      0.86      0.76        42
        184       0.88      0.88      0.88        58
        185       0.72      0.87      0.79        30
        186       0.96      0.95      0.95        56
        187       0.68      0.97      0.80        31
        188       0.97      1.00      0.99        33
        189       0.77      0.91      0.83        47
        190       0.95      0.93      0.94       108
        191       0.98      1.00      0.99        40
        192       0.97      0.94      0.95       119
        193       0.97      1.00      0.99        33
        194       0.77      0.81      0.79       152
        195       0.95      0.83      0.89        71
        196       0.92      0.92      0.92        39
        197       0.94      0.94      0.94       124
        198       0.86      0.77      0.81        31
        199       0.49      0.55      0.52        38
        200       0.94      0.97      0.96        34
        201       0.32      0.39      0.35        31
        202       0.82      0.85      0.83        53
        203       0.89      0.91      0.90        34
        204       0.59      0.82      0.69        39
        205       0.62      0.72      0.67        75
        206       0.98      0.88      0.93       101
        207       0.33      0.36      0.34        45
        208       0.81      0.71      0.76       108
        209       0.90      0.94      0.92        80
        210       0.82      0.85      0.84        33
        211       0.86      0.95      0.90        60
        212       0.74      0.68      0.71        93
        213       0.80      0.76      0.78        87
        214       0.25      0.22      0.24        36
        215       0.48      0.48      0.48        92
        216       0.75      0.83      0.78        75
        217       0.55      0.58      0.56        71
        218       0.52      0.62      0.57        56
        219       0.87      0.78      0.82        60
        220       0.81      0.86      0.83        50
        221       0.56      0.65      0.60        43
        222       1.00      1.00      1.00        46
        223       0.75      0.71      0.73        97
        224       1.00      0.84      0.91        85
        225       0.44      0.73      0.55        33
        226       0.91      1.00      0.96        43
        227       0.98      0.94      0.96        54
        228       0.52      0.80      0.63        30
        229       0.35      0.69      0.47        32
        230       0.58      0.59      0.59        37
        231       0.94      0.83      0.88        59
        232       0.92      0.86      0.89        98
        233       0.96      0.93      0.94        70
        234       0.96      0.97      0.96        69
        235       0.90      0.86      0.88       104
        236       1.00      0.96      0.98        99
        237       0.93      0.86      0.89       123
        238       0.91      0.90      0.91        93
        239       1.00      0.96      0.98        48
        240       0.58      0.63      0.60        82
        241       0.76      0.85      0.80        33
        242       0.82      0.85      0.84        95
        243       0.90      0.89      0.90        63
        244       0.42      0.54      0.47        41
        245       0.78      0.70      0.74       104
        246       0.83      0.69      0.75        77
        247       0.95      1.00      0.98        62
        248       0.97      0.95      0.96        78
        249       0.85      0.75      0.80        61
        250       0.91      0.85      0.88       121
        251       0.63      0.55      0.59        31
        252       0.74      0.93      0.82        42
        253       0.60      0.67      0.63        45
        254       0.23      0.31      0.27        48
        255       0.98      1.00      0.99        42
        256       1.00      0.98      0.99        98
        257       0.88      0.94      0.91        32
        258       0.46      0.53      0.49        55
        259       0.92      0.96      0.94       128
        260       0.65      0.49      0.56       121
        261       0.58      0.85      0.69        39
        262       0.98      1.00      0.99        62
        263       0.49      0.66      0.56        64
        264       0.93      0.93      0.93        45
        265       0.87      0.95      0.91        56
        266       0.90      0.94      0.92        68
        267       0.87      0.87      0.87        67
        268       0.84      0.80      0.82       148
        269       0.64      0.81      0.71        31
        270       0.63      0.42      0.50       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.94      1.00      0.97        44
        274       0.79      0.77      0.78       168
        275       0.31      0.63      0.41        38
        276       0.24      0.51      0.33        35
        277       0.87      0.95      0.91        58
        278       0.68      0.76      0.71        33
        279       0.54      0.57      0.55        63
        280       0.95      0.88      0.91        67
        281       0.64      0.84      0.73        45
        282       0.79      0.94      0.86        49
        283       0.82      1.00      0.90        32
        284       0.72      0.61      0.66        96
        285       0.77      0.80      0.78       118
        286       0.86      0.86      0.86        66
        287       0.77      0.67      0.72       107
        288       0.92      0.94      0.93        35
        289       0.93      0.88      0.90        88
        290       0.83      0.83      0.83        66
        291       0.99      0.88      0.93        96
        292       0.99      0.83      0.90       135
        293       0.69      0.84      0.76        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.64      0.59      0.61        82
        297       0.90      0.84      0.87       107
        298       0.74      0.76      0.75        41
        299       0.86      0.81      0.83        31
        300       0.77      0.88      0.82        42
        301       0.73      0.83      0.78        53
        302       0.95      0.93      0.94        90
        303       0.56      0.77      0.65        35
        304       0.91      0.98      0.94        42
        305       0.94      0.89      0.91       102
        306       1.00      0.95      0.97        39
        307       0.88      0.77      0.82       120
        308       0.96      0.99      0.97       117
        309       0.91      0.86      0.88        98
        310       0.96      0.99      0.98       110
        311       0.45      0.47      0.46        36
        312       0.96      0.99      0.98        99
        313       0.78      0.94      0.85       116
        314       0.59      0.74      0.66        39
        315       0.62      0.70      0.66        30
        316       0.93      0.83      0.88       156
        317       0.76      0.84      0.80        75
        318       0.65      0.74      0.69        72
        319       0.98      0.90      0.94       154
        320       0.95      0.85      0.90        86
        321       0.26      0.17      0.21        59
        322       0.74      0.80      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.88      0.91        56
        325       0.57      0.57      0.57       112
        326       0.92      0.90      0.91       114
        327       0.86      0.85      0.85        93
        328       0.75      0.72      0.73        85
        329       0.32      0.16      0.21        76
        330       0.64      0.67      0.65        66
        331       0.86      0.90      0.88       128
        332       0.88      0.91      0.90        33
        333       0.49      0.55      0.52        56
        334       0.62      0.44      0.51        41
        335       0.86      0.64      0.73       103
        336       0.80      0.85      0.82        46
        337       0.96      0.93      0.94        54
        338       0.79      0.92      0.85        52
        339       0.71      0.68      0.70        57
        340       0.84      0.76      0.80       128
        341       0.91      0.93      0.92        42
        342       0.79      0.66      0.72       118
        343       0.52      0.58      0.55        45
        344       0.60      0.74      0.66        38
        345       0.99      0.97      0.98       153
        346       0.60      0.78      0.68        65
        347       0.91      0.83      0.87        64
        348       0.75      0.56      0.64        99
        349       1.00      0.99      1.00       105
        350       0.78      0.80      0.79        86
        351       0.94      0.96      0.95       125
        352       0.42      0.36      0.39        64
        353       0.41      0.56      0.47        50
        354       0.41      0.68      0.51        41
        355       0.94      0.86      0.90       108
        356       0.80      0.75      0.77        48
        357       0.71      0.66      0.69        98
        358       0.64      0.63      0.64        43
        359       0.72      0.74      0.73       120
        360       1.00      0.99      1.00       103
        361       0.42      0.43      0.43       120
        362       0.98      0.89      0.93       158
        363       0.89      0.98      0.93        51
        364       0.56      0.76      0.65        42
        365       0.89      0.86      0.88       122
        366       0.71      0.58      0.64       110
        367       1.00      0.94      0.97        67
        368       0.86      0.86      0.86       114
        369       0.66      0.82      0.73        62
        370       0.91      0.93      0.92       130
        371       0.67      0.71      0.69        62
        372       0.81      0.83      0.82       110
        373       0.88      0.89      0.89       145
        374       0.46      0.75      0.57        32
        375       0.47      0.36      0.41       100
        376       0.95      0.97      0.96        39
        377       0.68      0.74      0.71        38
        378       0.37      0.58      0.45        31
        379       0.77      0.81      0.79        54
        380       0.72      0.89      0.80        44
        381       0.34      0.38      0.36        48
        382       0.70      0.84      0.76        31
        383       0.73      0.79      0.76        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.74      0.80      0.77        61
        387       0.59      0.75      0.66        32
        388       0.68      0.67      0.68       115
        389       0.77      0.86      0.81        43
        390       0.99      0.97      0.98       158
        391       0.65      0.72      0.68        89
        392       0.84      0.96      0.90        50
        393       0.93      0.93      0.93        69
        394       0.83      0.87      0.85        55
        395       0.80      0.86      0.83        81
        396       0.76      0.75      0.75        59
        397       0.94      0.97      0.95       121
        398       0.88      0.94      0.91        80
        399       0.81      0.87      0.84        45
        400       0.56      0.53      0.54        38
        401       0.77      0.84      0.81       108
        402       0.91      0.91      0.91        34
        403       0.90      0.84      0.87       123
        404       0.84      0.89      0.87        55
        405       0.87      0.84      0.86       103
        406       0.78      0.87      0.83       111
        407       0.72      0.76      0.74        76
        408       0.63      0.87      0.73        30
        409       0.56      0.60      0.57        42
        410       0.78      0.44      0.57        63
        411       0.83      0.76      0.79       136
        412       0.59      0.52      0.55        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.38      0.38        88
        416       0.89      0.70      0.78       116
        417       0.94      0.94      0.94       124
        418       0.91      0.72      0.81       127
        419       0.83      0.89      0.86        76
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.65      0.79      0.71        42
        423       0.75      0.70      0.72       110
        424       0.69      0.97      0.81        30
        425       0.75      0.89      0.81        37
        426       0.69      0.78      0.74        32
        427       0.75      0.94      0.83        32
        428       0.48      0.67      0.56        55
        429       0.96      0.98      0.97        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.77      0.65      0.70        71
        433       0.48      0.51      0.50        55
        434       0.91      0.94      0.92        32
        435       0.92      0.82      0.86        66
        436       0.90      0.88      0.89        65
        437       0.75      0.94      0.83        32
        438       0.77      0.67      0.71        54
        439       0.78      0.70      0.74        30
        440       0.98      0.99      0.98        86
        441       0.38      0.64      0.48        33
        442       0.79      0.76      0.78       131
        443       0.53      0.64      0.58        86
        444       0.50      0.46      0.48        48
        445       0.75      0.95      0.84        38
        446       0.97      0.99      0.98       123
        447       0.60      0.71      0.65        63
        448       0.86      0.76      0.81        58
        449       0.84      0.95      0.89        57
        450       0.59      0.71      0.65        56
        451       0.70      0.78      0.74        51
        452       0.53      0.73      0.61        41
        453       0.96      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.79      0.87      0.83        63
        456       0.92      0.81      0.86        85
        457       0.91      0.78      0.84       137
        458       0.75      0.81      0.78        47
        459       0.60      0.38      0.46        87
        460       0.99      0.99      0.99        95
        461       0.51      0.76      0.61        46
        462       0.56      0.52      0.54        69
        463       0.88      0.71      0.78       120
        464       0.87      0.83      0.85        93
        465       0.73      0.89      0.80        63
        466       0.41      0.56      0.47        32
        467       0.77      0.70      0.73        73
        468       0.93      0.98      0.95        42
        469       0.91      0.87      0.89        78
        470       1.00      0.97      0.99        36
        471       0.81      0.66      0.73        44
        472       0.67      0.66      0.67       112
        473       0.85      0.94      0.89        36
        474       0.54      0.47      0.50        79
        475       0.46      0.74      0.57        31
        476       0.99      0.90      0.94        78
        477       0.93      0.83      0.88       107
        478       0.90      0.58      0.70        33
        479       0.88      0.94      0.91        63
        480       0.91      0.95      0.93        91
        481       0.71      0.76      0.74        46
        482       0.91      0.88      0.89        33
        483       0.99      0.86      0.92       167
        484       1.00      0.97      0.99        35
        485       0.82      0.75      0.78        36
        486       0.29      0.30      0.29        74
        487       0.74      0.77      0.75        60
        488       0.55      0.68      0.61        31
        489       0.92      0.92      0.92        49
        490       0.82      0.65      0.73       121
        491       0.80      0.85      0.83        39
        492       0.74      0.75      0.74       114
        493       0.95      0.85      0.89       149
        494       0.66      0.75      0.70        51
        495       0.93      0.93      0.93        80
        496       0.81      0.97      0.88        39
        497       0.60      0.48      0.53        88

avg / total       0.80      0.79      0.79     35746

Iteration number, batch number :  7 0
Training data accuracy :  0.841365461847
Training data loss     :  0.00516192505987
Iteration number, batch number :  7 1
Training data accuracy :  0.826305220884
Training data loss     :  0.00524982583117
Iteration number, batch number :  7 2
Training data accuracy :  0.832329317269
Training data loss     :  0.00522001006281
Iteration number, batch number :  7 3
Training data accuracy :  0.841365461847
Training data loss     :  0.00522166875617
Iteration number, batch number :  7 4
Training data accuracy :  0.841365461847
Training data loss     :  0.00519789472667
Iteration number, batch number :  7 5
Training data accuracy :  0.860441767068
Training data loss     :  0.00499434172873
Iteration number, batch number :  7 6
Training data accuracy :  0.843373493976
Training data loss     :  0.00537795463887
Iteration number, batch number :  7 7
Training data accuracy :  0.85843373494
Training data loss     :  0.00457320294516
Iteration number, batch number :  7 8
Training data accuracy :  0.846385542169
Training data loss     :  0.00500246987535
Iteration number, batch number :  7 9
Training data accuracy :  0.848393574297
Training data loss     :  0.00507532003528
Iteration number, batch number :  7 10
Training data accuracy :  0.867469879518
Training data loss     :  0.00450715543698
Iteration number, batch number :  7 11
Training data accuracy :  0.845381526104
Training data loss     :  0.00490924131677
Iteration number, batch number :  7 12
Training data accuracy :  0.864457831325
Training data loss     :  0.0048341776883
Iteration number, batch number :  7 13
Training data accuracy :  0.84437751004
Training data loss     :  0.00531871243305
Iteration number, batch number :  7 14
Training data accuracy :  0.85843373494
Training data loss     :  0.00465727473783
Iteration number, batch number :  7 15
Training data accuracy :  0.853413654618
Training data loss     :  0.00457547621772
Iteration number, batch number :  7 16
Training data accuracy :  0.876506024096
Training data loss     :  0.00429536558192
Iteration number, batch number :  7 17
Training data accuracy :  0.856425702811
Training data loss     :  0.00457193466901
Iteration number, batch number :  7 18
Training data accuracy :  0.853413654618
Training data loss     :  0.00496472350837
Iteration number, batch number :  7 19
Training data accuracy :  0.863453815261
Training data loss     :  0.00478754622459
Iteration number, batch number :  7 20
Training data accuracy :  0.868473895582
Training data loss     :  0.00452031326763
Iteration number, batch number :  7 21
Training data accuracy :  0.84437751004
Training data loss     :  0.00460207254275
Iteration number, batch number :  7 22
Training data accuracy :  0.863453815261
Training data loss     :  0.00436012065467
Iteration number, batch number :  7 23
Training data accuracy :  0.855421686747
Training data loss     :  0.00433113334426
Iteration number, batch number :  7 24
Training data accuracy :  0.848393574297
Training data loss     :  0.0048114913123
Iteration number, batch number :  7 25
Training data accuracy :  0.852409638554
Training data loss     :  0.00488700341821
Iteration number, batch number :  7 26
Training data accuracy :  0.869477911647
Training data loss     :  0.00460057027658
Iteration number, batch number :  7 27
Training data accuracy :  0.855421686747
Training data loss     :  0.00466951782334
Iteration number, batch number :  7 28
Training data accuracy :  0.875502008032
Training data loss     :  0.00425062689028
Iteration number, batch number :  7 29
Training data accuracy :  0.856425702811
Training data loss     :  0.00472661526128
Iteration number, batch number :  7 30
Training data accuracy :  0.852409638554
Training data loss     :  0.00456596681943
Iteration number, batch number :  7 31
Training data accuracy :  0.846385542169
Training data loss     :  0.00482722457985
Iteration number, batch number :  7 32
Training data accuracy :  0.856425702811
Training data loss     :  0.00469215714635
Iteration number, batch number :  7 33
Training data accuracy :  0.887550200803
Training data loss     :  0.00431944562315
Iteration number, batch number :  7 34
Training data accuracy :  0.882530120482
Training data loss     :  0.00418746115641
Iteration number, batch number :  7 35
Training data accuracy :  0.875502008032
Training data loss     :  0.00404104070107
Iteration number, batch number :  7 36
Training data accuracy :  0.862449799197
Training data loss     :  0.00428640844043
Iteration number, batch number :  7 37
Training data accuracy :  0.882530120482
Training data loss     :  0.00435861375367
Iteration number, batch number :  7 38
Training data accuracy :  0.873493975904
Training data loss     :  0.00439082301811
Iteration number, batch number :  7 39
Training data accuracy :  0.893574297189
Training data loss     :  0.00386689436558
Iteration number, batch number :  7 40
Training data accuracy :  0.85843373494
Training data loss     :  0.00474445935997
Iteration number, batch number :  7 41
Training data accuracy :  0.862449799197
Training data loss     :  0.00433494099517
Iteration number, batch number :  7 42
Training data accuracy :  0.880522088353
Training data loss     :  0.004306611459
Iteration number, batch number :  7 43
Training data accuracy :  0.875502008032
Training data loss     :  0.00432228283734
Iteration number, batch number :  7 44
Training data accuracy :  0.871485943775
Training data loss     :  0.00393005947225
Iteration number, batch number :  7 45
Training data accuracy :  0.876506024096
Training data loss     :  0.00417295431506
Iteration number, batch number :  7 46
Training data accuracy :  0.869477911647
Training data loss     :  0.00411594563254
Iteration number, batch number :  7 47
Training data accuracy :  0.877510040161
Training data loss     :  0.00445420202602
Iteration number, batch number :  7 48
Training data accuracy :  0.857429718876
Training data loss     :  0.00460816837871
Iteration number, batch number :  7 49
Training data accuracy :  0.857429718876
Training data loss     :  0.00430445376485
Iteration number, batch number :  7 50
Training data accuracy :  0.871485943775
Training data loss     :  0.00434297369743
Iteration number, batch number :  7 51
Training data accuracy :  0.84437751004
Training data loss     :  0.00468361936748
Iteration number, batch number :  7 52
Training data accuracy :  0.855421686747
Training data loss     :  0.0051776831532
Iteration number, batch number :  7 53
Training data accuracy :  0.850401606426
Training data loss     :  0.00487867694582
Iteration number, batch number :  7 54
Training data accuracy :  0.856425702811
Training data loss     :  0.00486519786374
Iteration number, batch number :  7 55
Training data accuracy :  0.849397590361
Training data loss     :  0.00471796495791
Iteration number, batch number :  7 56
Training data accuracy :  0.859437751004
Training data loss     :  0.00494466299533
Iteration number, batch number :  7 57
Training data accuracy :  0.84437751004
Training data loss     :  0.00505082440796
Iteration number, batch number :  7 58
Training data accuracy :  0.83734939759
Training data loss     :  0.00503293499078
Iteration number, batch number :  7 59
Training data accuracy :  0.848393574297
Training data loss     :  0.00527753648202
Iteration number, batch number :  7 60
Training data accuracy :  0.850401606426
Training data loss     :  0.00496143629213
Iteration number, batch number :  7 61
Training data accuracy :  0.841365461847
Training data loss     :  0.00515472541443
Iteration number, batch number :  7 62
Training data accuracy :  0.840361445783
Training data loss     :  0.00508124992675
Iteration number, batch number :  7 63
Training data accuracy :  0.843373493976
Training data loss     :  0.00520329050883
Iteration number, batch number :  7 64
Training data accuracy :  0.849397590361
Training data loss     :  0.00546288978615
Iteration number, batch number :  7 65
Training data accuracy :  0.839357429719
Training data loss     :  0.00563788541536
Iteration number, batch number :  7 66
Training data accuracy :  0.84437751004
Training data loss     :  0.00495740416612
Iteration number, batch number :  7 67
Training data accuracy :  0.856425702811
Training data loss     :  0.00513081315142
Iteration number, batch number :  7 68
Training data accuracy :  0.847389558233
Training data loss     :  0.00535942645117
Iteration number, batch number :  7 69
Training data accuracy :  0.841365461847
Training data loss     :  0.00507223627987
Iteration number, batch number :  7 35
Iteration number, batch number :  7 34
Iteration number, batch number :  7 33
Iteration number, batch number :  7 32
Iteration number, batch number :  7 31
Iteration number, batch number :  7 30
Iteration number, batch number :  7 29
Iteration number, batch number :  7 28
Iteration number, batch number :  7 27
Iteration number, batch number :  7 26
Iteration number, batch number :  7 25
Iteration number, batch number :  7 24
Iteration number, batch number :  7 23
Iteration number, batch number :  7 22
Iteration number, batch number :  7 21
Iteration number, batch number :  7 20
Iteration number, batch number :  7 19
Iteration number, batch number :  7 18
Iteration number, batch number :  7 17
Iteration number, batch number :  7 16
Iteration number, batch number :  7 15
Iteration number, batch number :  7 14
Iteration number, batch number :  7 13
Iteration number, batch number :  7 12
Iteration number, batch number :  7 11
Iteration number, batch number :  7 10
Iteration number, batch number :  7 9
Iteration number, batch number :  7 8
Iteration number, batch number :  7 7
Iteration number, batch number :  7 6
Iteration number, batch number :  7 5
Iteration number, batch number :  7 4
Iteration number, batch number :  7 3
Iteration number, batch number :  7 2
Iteration number, batch number :  7 1
Iteration number, batch number :  7 0
Accuracy on test data :  76.5778203539 27179.0 27179
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.77      0.91      0.84        45
          1       0.74      0.95      0.83        41
          2       0.64      0.81      0.71        37
          3       0.82      0.96      0.88        52
          4       0.95      0.55      0.70       130
          5       0.92      0.97      0.95        36
          6       0.50      0.71      0.59        51
          7       0.71      0.62      0.66        47
          8       0.39      0.38      0.38        76
          9       0.96      1.00      0.98        51
         10       0.95      0.93      0.94        45
         11       0.36      0.45      0.40        31
         12       0.93      0.86      0.89        79
         13       0.71      0.82      0.77        85
         14       0.92      0.86      0.89       143
         15       0.50      0.42      0.45       120
         16       0.91      0.94      0.93        34
         17       0.65      0.31      0.42       124
         18       0.77      0.81      0.79        70
         19       0.54      0.55      0.54        91
         20       0.41      0.62      0.49        39
         21       0.90      0.34      0.49        53
         22       0.53      0.51      0.52        45
         23       0.48      0.73      0.58        33
         24       0.79      0.78      0.78        40
         25       0.97      0.98      0.97        85
         26       0.93      0.73      0.82       132
         27       0.92      0.88      0.90        90
         28       0.36      0.53      0.43        30
         29       0.84      0.84      0.84        96
         30       0.90      0.86      0.88        74
         31       0.97      0.89      0.93        79
         32       1.00      0.99      0.99       269
         33       0.97      0.94      0.95        31
         34       0.40      0.66      0.49        32
         35       0.87      0.94      0.90        77
         36       0.87      0.92      0.90        66
         37       0.76      0.84      0.80       114
         38       0.05      0.01      0.01       121
         39       0.76      0.85      0.81        34
         40       0.34      0.45      0.39        76
         41       0.92      0.64      0.76        92
         42       0.66      0.72      0.69        58
         43       0.93      0.87      0.90        63
         44       0.57      0.80      0.67        30
         45       0.81      0.95      0.87        40
         46       0.00      0.00      0.00       161
         47       0.58      0.79      0.67        47
         48       0.92      0.94      0.93        36
         49       0.83      0.95      0.89        37
         50       0.80      0.65      0.72       123
         51       0.60      0.47      0.53       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.94      0.90      0.92        70
         55       0.99      0.94      0.96        94
         56       0.60      0.65      0.62        66
         57       0.68      0.67      0.68        79
         58       0.97      0.92      0.94        98
         59       0.95      0.90      0.93        92
         60       1.00      0.88      0.94        33
         61       0.86      0.93      0.89        41
         62       0.33      0.07      0.11        30
         63       0.84      0.88      0.86        74
         64       0.88      1.00      0.94        38
         65       0.95      0.88      0.92        95
         66       0.80      0.70      0.75        94
         67       0.88      0.91      0.90       132
         68       0.36      0.42      0.38        72
         69       0.34      0.50      0.41        30
         70       0.74      0.81      0.77        42
         71       0.67      0.82      0.73        44
         72       0.34      0.67      0.45        42
         73       0.82      0.84      0.83        37
         74       0.64      0.89      0.74        98
         75       0.95      0.87      0.91       163
         76       0.91      0.91      0.91        65
         77       0.97      0.94      0.96        70
         78       0.85      0.85      0.85        33
         79       0.60      0.82      0.69        33
         80       0.77      0.66      0.71        71
         81       0.96      0.96      0.96       131
         82       0.84      0.70      0.76       174
         83       0.71      0.74      0.73        47
         84       0.23      0.27      0.25        44
         85       0.96      0.93      0.94       110
         86       0.54      0.45      0.49        97
         87       0.92      0.78      0.84        83
         88       0.75      0.84      0.79        43
         89       0.73      0.91      0.81        35
         90       0.72      0.82      0.77        38
         91       0.70      0.61      0.65        51
         92       0.44      0.72      0.55        40
         93       0.94      0.92      0.93        48
         94       0.90      0.90      0.90        71
         95       0.84      0.79      0.81       117
         96       0.55      0.84      0.67        38
         97       0.69      0.90      0.78        30
         98       0.88      0.85      0.86        98
         99       0.60      0.75      0.67        69
        100       0.92      0.91      0.91        98
        101       0.64      0.65      0.64        85
        102       0.40      0.57      0.47        56
        103       0.97      0.92      0.94        36
        104       0.69      0.60      0.64        40
        105       0.69      0.55      0.62        65
        106       0.63      0.75      0.69        32
        107       1.00      0.53      0.69        34
        108       0.69      0.75      0.72        81
        109       0.61      0.22      0.32        77
        110       0.96      0.91      0.93       121
        111       0.58      0.57      0.57        53
        112       0.91      0.89      0.90        88
        113       0.78      0.97      0.87        30
        114       0.84      0.88      0.86       127
        115       0.92      0.73      0.81        78
        116       0.59      0.53      0.56       108
        117       0.93      0.83      0.87       110
        118       0.53      0.41      0.46        88
        119       0.90      0.80      0.85       106
        120       0.63      0.75      0.69        69
        121       0.79      0.85      0.81        39
        122       0.88      0.77      0.82       124
        123       0.86      0.83      0.84        59
        124       1.00      0.96      0.98        69
        125       0.73      0.87      0.80        38
        126       0.77      0.66      0.71       113
        127       0.98      1.00      0.99        55
        128       0.71      0.85      0.78        82
        129       0.93      0.89      0.91       125
        130       0.58      0.83      0.68        47
        131       0.83      0.36      0.50       112
        132       0.99      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.47      0.62      0.53        32
        135       0.81      0.82      0.82        80
        136       0.74      0.71      0.72       103
        137       0.76      0.81      0.79       134
        138       0.92      0.91      0.91        88
        139       0.17      0.08      0.11        37
        140       0.92      1.00      0.96        44
        141       0.84      0.92      0.88        61
        142       0.98      0.94      0.96       103
        143       0.54      0.69      0.60        51
        144       0.48      0.65      0.55        72
        145       0.72      0.96      0.82        49
        146       0.77      0.76      0.77        85
        147       0.91      0.87      0.89        93
        148       0.86      0.86      0.86        57
        149       0.84      0.86      0.85       109
        150       0.88      0.64      0.74        76
        151       0.68      0.68      0.68        75
        152       0.94      0.91      0.92        54
        153       0.45      0.78      0.57        32
        154       0.90      0.88      0.89        73
        155       0.85      0.61      0.71       189
        156       0.66      0.49      0.56        92
        157       0.94      0.97      0.95        31
        158       0.47      0.15      0.23       100
        159       0.51      0.42      0.46       108
        160       0.23      0.44      0.30        39
        161       0.68      0.74      0.71        57
        162       0.92      1.00      0.96        33
        163       0.39      0.47      0.43        70
        164       0.96      0.81      0.88       159
        165       0.48      0.72      0.58        39
        166       0.94      0.94      0.94        71
        167       0.85      0.80      0.82        64
        168       0.85      0.81      0.83        75
        169       0.53      0.64      0.58        39
        170       0.99      0.94      0.97       140
        171       0.17      0.08      0.11        61
        172       0.78      0.90      0.84        88
        173       0.63      0.69      0.66        39
        174       0.82      0.89      0.86        37
        175       0.77      0.82      0.80        45
        176       0.36      0.50      0.42        42
        177       0.61      0.87      0.72        31
        178       0.93      0.95      0.94        55
        179       0.35      0.58      0.44        31
        180       0.65      0.62      0.63        82
        181       0.57      0.65      0.61        31
        182       0.71      0.67      0.69        87
        183       0.76      0.90      0.82        41
        184       0.88      0.91      0.90        57
        185       0.76      0.87      0.81        30
        186       1.00      0.95      0.97        56
        187       0.76      0.97      0.85        30
        188       0.97      1.00      0.99        33
        189       0.71      0.78      0.74        46
        190       0.91      0.99      0.95       107
        191       0.93      0.95      0.94        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.80      0.74      0.77       152
        195       0.91      0.85      0.88        71
        196       0.97      0.95      0.96        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.39      0.70      0.50        37
        200       0.97      1.00      0.99        34
        201       0.16      0.27      0.20        30
        202       0.70      0.81      0.75        53
        203       0.86      0.97      0.91        33
        204       0.69      0.92      0.79        38
        205       0.52      0.55      0.53        75
        206       0.95      0.93      0.94       101
        207       0.27      0.27      0.27        45
        208       0.79      0.77      0.78       108
        209       0.79      0.85      0.82        79
        210       0.73      0.94      0.82        32
        211       0.92      0.90      0.91        60
        212       0.74      0.72      0.73        93
        213       0.77      0.79      0.78        87
        214       0.23      0.34      0.28        35
        215       0.56      0.56      0.56        91
        216       0.80      0.53      0.64        75
        217       0.54      0.61      0.58        70
        218       0.56      0.69      0.62        55
        219       0.92      0.73      0.81        60
        220       0.82      0.92      0.87        50
        221       0.53      0.69      0.60        42
        222       1.00      0.96      0.98        45
        223       0.80      0.74      0.77        96
        224       0.91      0.85      0.88        85
        225       0.54      0.84      0.66        32
        226       0.95      0.98      0.97        43
        227       0.90      0.98      0.94        54
        228       0.43      0.87      0.58        30
        229       0.27      0.50      0.35        32
        230       0.52      0.62      0.57        37
        231       0.80      0.90      0.85        59
        232       0.96      0.84      0.90        97
        233       0.97      0.94      0.96        69
        234       0.93      0.57      0.70        69
        235       0.86      0.76      0.80       103
        236       0.99      0.98      0.98        98
        237       0.89      0.71      0.79       123
        238       0.85      0.90      0.88        93
        239       1.00      0.96      0.98        48
        240       0.61      0.65      0.63        81
        241       0.86      0.94      0.90        32
        242       0.88      0.75      0.81        95
        243       0.81      0.87      0.84        63
        244       0.44      0.60      0.51        40
        245       0.79      0.77      0.78       103
        246       0.70      0.52      0.60        77
        247       0.94      1.00      0.97        61
        248       0.95      0.90      0.92        78
        249       0.91      0.83      0.87        60
        250       0.94      0.93      0.94       120
        251       0.56      0.73      0.64        30
        252       0.82      0.86      0.84        42
        253       0.55      0.64      0.59        44
        254       0.25      0.34      0.29        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.81      0.94      0.87        31
        258       0.38      0.33      0.35        55
        259       0.96      0.92      0.94       128
        260       0.61      0.42      0.50       121
        261       0.67      0.85      0.75        39
        262       0.97      0.98      0.98        62
        263       0.49      0.70      0.58        63
        264       0.87      0.89      0.88        45
        265       0.71      0.91      0.80        55
        266       0.91      0.94      0.93        67
        267       0.86      0.77      0.82        66
        268       0.80      0.70      0.74       148
        269       0.52      0.90      0.66        30
        270       0.59      0.46      0.52       149
        271       0.80      0.93      0.86        42
        272       0.97      0.95      0.96       119
        273       0.91      1.00      0.96        43
        274       0.80      0.75      0.78       167
        275       0.35      0.54      0.43        37
        276       0.19      0.40      0.26        35
        277       0.93      0.96      0.95        57
        278       0.41      0.28      0.33        32
        279       0.47      0.53      0.50        62
        280       0.00      0.00      0.00        66
        281       0.68      0.89      0.77        45
        282       0.86      0.90      0.88        48
        283       0.79      0.97      0.87        31
        284       0.71      0.66      0.68        96
        285       0.78      0.75      0.77       117
        286       0.97      0.85      0.90        66
        287       0.73      0.72      0.73       107
        288       0.69      0.89      0.78        35
        289       0.96      0.92      0.94        87
        290       0.72      0.73      0.72        66
        291       0.96      0.89      0.92        96
        292       0.97      0.70      0.82       135
        293       0.61      0.91      0.73        43
        294       0.47      0.90      0.62        31
        295       0.77      0.86      0.81        56
        296       0.48      0.53      0.51        81
        297       0.89      0.89      0.89       107
        298       0.68      0.68      0.68        41
        299       0.67      0.90      0.77        31
        300       0.79      0.80      0.80        41
        301       0.69      0.81      0.74        52
        302       0.80      0.91      0.85        90
        303       0.49      0.82      0.62        34
        304       0.91      0.95      0.93        41
        305       0.96      0.87      0.91       101
        306       0.95      0.95      0.95        38
        307       0.90      0.67      0.77       120
        308       0.97      0.96      0.96       117
        309       0.90      0.86      0.88        97
        310       0.95      0.95      0.95       109
        311       0.36      0.46      0.40        35
        312       0.95      0.90      0.92        98
        313       0.67      0.92      0.78       116
        314       0.48      0.72      0.58        39
        315       0.71      0.83      0.77        30
        316       0.95      0.91      0.93       155
        317       0.75      0.85      0.80        75
        318       0.71      0.62      0.67        72
        319       0.98      0.86      0.91       154
        320       0.85      0.84      0.84        85
        321       0.04      0.02      0.02        58
        322       0.77      0.81      0.79       119
        323       1.00      1.00      1.00        43
        324       0.95      0.96      0.95        55
        325       0.61      0.59      0.60       112
        326       0.91      0.87      0.89       114
        327       0.86      0.80      0.83        92
        328       0.71      0.71      0.71        84
        329       0.31      0.16      0.21        75
        330       0.67      0.68      0.68        66
        331       0.93      0.87      0.90       127
        332       0.84      0.94      0.89        33
        333       0.50      0.58      0.54        55
        334       0.50      0.29      0.37        41
        335       0.82      0.79      0.80       103
        336       0.74      0.89      0.81        45
        337       0.89      0.93      0.91        54
        338       0.80      0.88      0.84        51
        339       0.69      0.75      0.72        57
        340       0.79      0.76      0.77       127
        341       0.93      0.95      0.94        41
        342       0.75      0.71      0.73       118
        343       0.58      0.67      0.62        45
        344       0.62      0.89      0.73        37
        345       0.99      0.97      0.98       152
        346       0.58      0.69      0.63        64
        347       0.89      0.90      0.90        63
        348       0.73      0.58      0.65        98
        349       0.99      0.98      0.99       104
        350       0.77      0.85      0.81        86
        351       0.88      0.86      0.87       124
        352       0.45      0.60      0.51        63
        353       0.53      0.52      0.53        50
        354       0.49      0.70      0.58        40
        355       0.95      0.87      0.91       108
        356       0.82      0.89      0.86        47
        357       0.70      0.58      0.63        98
        358       0.65      0.81      0.72        42
        359       0.75      0.83      0.79       120
        360       1.00      0.92      0.96       102
        361       0.50      0.44      0.47       120
        362       0.94      0.93      0.94       157
        363       0.91      0.98      0.94        51
        364       0.46      0.67      0.54        42
        365       0.94      0.89      0.91       122
        366       0.69      0.61      0.64       109
        367       1.00      0.97      0.98        67
        368       0.83      0.86      0.84       113
        369       0.55      0.71      0.62        62
        370       0.98      0.92      0.95       130
        371       0.58      0.72      0.64        61
        372       0.81      0.83      0.82       109
        373       0.83      0.80      0.81       145
        374       0.47      0.71      0.56        31
        375       0.52      0.34      0.41        99
        376       0.90      0.92      0.91        38
        377       0.74      0.78      0.76        37
        378       0.25      0.65      0.36        31
        379       0.76      0.81      0.79        54
        380       0.73      0.91      0.81        44
        381       0.23      0.38      0.28        48
        382       0.71      0.94      0.81        31
        383       0.69      0.74      0.71        89
        384       0.99      1.00      1.00       100
        385       0.98      0.89      0.93        45
        386       0.82      0.83      0.83        60
        387       0.62      0.77      0.69        31
        388       0.68      0.74      0.71       115
        389       0.74      0.86      0.80        43
        390       0.99      0.99      0.99       158
        391       0.69      0.57      0.63        89
        392       0.84      0.94      0.89        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.73      0.80      0.76        80
        396       0.63      0.74      0.68        58
        397       0.96      0.93      0.94       121
        398       0.90      0.94      0.92        80
        399       0.95      0.93      0.94        44
        400       0.45      0.68      0.54        37
        401       0.85      0.76      0.80       108
        402       0.94      0.88      0.91        33
        403       0.93      0.88      0.90       123
        404       0.84      0.95      0.89        55
        405       0.90      0.72      0.80       102
        406       0.83      0.83      0.83       110
        407       0.61      0.62      0.61        76
        408       0.65      0.93      0.77        30
        409       0.48      0.60      0.53        42
        410       0.78      0.40      0.53        62
        411       0.74      0.71      0.72       135
        412       0.50      0.53      0.52        64
        413       0.97      0.97      0.97        31
        414       0.95      0.92      0.94        66
        415       0.29      0.26      0.27        88
        416       0.85      0.82      0.83       115
        417       0.97      0.95      0.96       124
        418       0.90      0.77      0.83       126
        419       0.78      0.47      0.58        75
        420       0.86      0.92      0.89        48
        421       0.88      1.00      0.94        30
        422       0.52      0.81      0.63        42
        423       0.66      0.67      0.67       109
        424       0.91      1.00      0.95        30
        425       0.76      0.92      0.83        37
        426       0.45      0.65      0.53        31
        427       0.70      0.97      0.82        32
        428       0.53      0.70      0.60        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       0.98      0.98      0.98        42
        432       0.67      0.69      0.68        70
        433       0.39      0.49      0.43        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.54      0.78      0.64        64
        437       0.70      0.88      0.78        32
        438       0.82      0.68      0.74        53
        439       0.66      0.70      0.68        30
        440       0.96      0.96      0.96        85
        441       0.36      0.64      0.46        33
        442       0.78      0.81      0.79       131
        443       0.54      0.61      0.57        85
        444       0.55      0.70      0.62        47
        445       0.82      0.87      0.85        38
        446       0.96      0.86      0.91       122
        447       0.61      0.49      0.54        63
        448       0.90      0.91      0.90        57
        449       0.89      1.00      0.94        57
        450       0.56      0.73      0.64        56
        451       0.62      0.71      0.66        51
        452       0.45      0.50      0.48        40
        453       0.95      0.91      0.93       137
        454       0.97      0.92      0.95        39
        455       0.73      0.87      0.80        63
        456       0.92      0.93      0.92        84
        457       0.91      0.78      0.84       136
        458       0.57      0.96      0.71        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.58        46
        462       0.52      0.49      0.50        68
        463       0.79      0.81      0.80       120
        464       0.89      0.83      0.86        93
        465       0.63      0.84      0.72        62
        466       0.46      0.74      0.57        31
        467       0.75      0.69      0.72        72
        468       0.89      0.98      0.93        41
        469       0.97      0.88      0.93        78
        470       0.97      0.92      0.94        36
        471       0.40      0.05      0.08        44
        472       0.63      0.71      0.67       112
        473       0.92      0.94      0.93        35
        474       0.49      0.50      0.49        78
        475       0.58      0.83      0.68        30
        476       0.91      0.86      0.88        78
        477       0.88      0.79      0.84       106
        478       0.75      0.47      0.58        32
        479       0.78      0.90      0.84        63
        480       0.93      0.98      0.95        91
        481       0.78      0.84      0.81        45
        482       0.91      0.91      0.91        32
        483       0.99      0.92      0.95       166
        484       0.94      1.00      0.97        34
        485       0.63      0.91      0.74        35
        486       0.33      0.33      0.33        73
        487       0.62      0.64      0.63        59
        488       0.49      0.87      0.63        31
        489       0.78      0.94      0.85        48
        490       0.63      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.78      0.81       114
        493       0.96      0.89      0.92       149
        494       0.65      0.67      0.66        51
        495       0.90      0.88      0.89        80
        496       0.93      1.00      0.96        38
        497       0.48      0.52      0.50        88

avg / total       0.77      0.77      0.76     35492

Iteration number, batch number :  7 35
Iteration number, batch number :  7 34
Iteration number, batch number :  7 33
Iteration number, batch number :  7 32
Iteration number, batch number :  7 31
Iteration number, batch number :  7 30
Iteration number, batch number :  7 29
Iteration number, batch number :  7 28
Iteration number, batch number :  7 27
Iteration number, batch number :  7 26
Iteration number, batch number :  7 25
Iteration number, batch number :  7 24
Iteration number, batch number :  7 23
Iteration number, batch number :  7 22
Iteration number, batch number :  7 21
Iteration number, batch number :  7 20
Iteration number, batch number :  7 19
Iteration number, batch number :  7 18
Iteration number, batch number :  7 17
Iteration number, batch number :  7 16
Iteration number, batch number :  7 15
Iteration number, batch number :  7 14
Iteration number, batch number :  7 13
Iteration number, batch number :  7 12
Iteration number, batch number :  7 11
Iteration number, batch number :  7 10
Iteration number, batch number :  7 9
Iteration number, batch number :  7 8
Iteration number, batch number :  7 7
Iteration number, batch number :  7 6
Iteration number, batch number :  7 5
Iteration number, batch number :  7 4
Iteration number, batch number :  7 3
Iteration number, batch number :  7 2
Iteration number, batch number :  7 1
Iteration number, batch number :  7 0
Accuracy on cv data :  79.5949197113 28452.0 28452
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.75      0.90      0.82        42
          2       0.67      0.92      0.77        37
          3       0.95      1.00      0.97        53
          4       0.97      0.85      0.91       131
          5       0.87      0.92      0.89        37
          6       0.48      0.71      0.57        51
          7       0.68      0.57      0.62        47
          8       0.61      0.55      0.58        77
          9       0.98      1.00      0.99        51
         10       0.93      0.93      0.93        46
         11       0.76      0.88      0.81        32
         12       0.87      0.94      0.90        79
         13       0.77      0.89      0.83        85
         14       0.93      0.97      0.95       143
         15       0.48      0.38      0.42       121
         16       0.85      0.94      0.89        35
         17       0.84      0.74      0.79       125
         18       0.79      0.83      0.81        70
         19       0.65      0.76      0.70        92
         20       0.68      0.77      0.72        39
         21       0.96      0.96      0.96        53
         22       0.73      0.72      0.73        46
         23       0.59      0.91      0.71        33
         24       0.77      0.90      0.83        40
         25       0.99      0.98      0.98        86
         26       0.96      0.90      0.93       132
         27       0.80      0.87      0.83        90
         28       0.55      0.84      0.67        31
         29       0.89      0.88      0.88        96
         30       0.89      0.89      0.89        74
         31       0.99      0.97      0.98        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.61      0.78      0.68        32
         35       0.88      0.92      0.90        77
         36       0.90      0.95      0.93        66
         37       0.75      0.77      0.76       115
         38       0.14      0.04      0.06       121
         39       0.84      0.91      0.88        35
         40       0.36      0.38      0.37        76
         41       0.89      0.83      0.86        93
         42       0.68      0.83      0.75        59
         43       0.92      0.95      0.94        63
         44       0.68      0.84      0.75        31
         45       0.86      0.93      0.89        40
         46       0.86      0.60      0.71       161
         47       0.61      0.77      0.68        48
         48       0.91      0.84      0.87        37
         49       0.76      0.89      0.82        38
         50       0.83      0.73      0.77       124
         51       0.54      0.41      0.46       101
         52       0.96      0.96      0.96        51
         53       0.97      0.99      0.98        77
         54       0.94      0.94      0.94        71
         55       1.00      0.96      0.98        95
         56       0.57      0.53      0.55        66
         57       0.80      0.75      0.77        80
         58       0.98      0.99      0.98        99
         59       0.97      0.90      0.93        93
         60       0.97      0.94      0.96        34
         61       0.85      1.00      0.92        41
         62       0.79      0.84      0.81        31
         63       0.99      0.92      0.95        74
         64       0.82      0.97      0.89        38
         65       0.90      0.88      0.89        96
         66       0.81      0.87      0.84        95
         67       0.88      0.92      0.90       132
         68       0.55      0.55      0.55        73
         69       0.45      0.43      0.44        30
         70       0.79      0.88      0.84        43
         71       0.63      0.66      0.64        44
         72       0.45      0.62      0.52        42
         73       0.91      0.84      0.87        37
         74       0.71      0.86      0.78        99
         75       0.90      0.87      0.88       164
         76       0.94      0.95      0.95        65
         77       0.94      0.96      0.95        71
         78       0.93      0.76      0.84        34
         79       0.70      0.85      0.77        33
         80       0.76      0.86      0.81        72
         81       0.93      0.98      0.95       132
         82       0.87      0.63      0.73       175
         83       0.78      0.73      0.75        48
         84       0.41      0.42      0.42        45
         85       1.00      0.96      0.98       110
         86       0.65      0.56      0.60        98
         87       0.80      0.73      0.77        83
         88       0.83      0.88      0.85        43
         89       0.84      0.91      0.88        35
         90       0.78      0.82      0.79        38
         91       0.64      0.53      0.58        51
         92       0.31      0.49      0.38        41
         93       0.86      0.92      0.89        48
         94       0.96      0.92      0.94        71
         95       0.83      0.82      0.83       117
         96       0.69      0.89      0.78        38
         97       0.66      0.81      0.72        31
         98       0.95      0.88      0.91        99
         99       0.76      0.75      0.76        69
        100       0.90      0.90      0.90        99
        101       0.66      0.61      0.63        85
        102       0.57      0.66      0.61        56
        103       0.94      0.94      0.94        36
        104       0.72      0.76      0.74        41
        105       0.72      0.65      0.68        65
        106       0.70      0.81      0.75        32
        107       0.93      0.74      0.83        35
        108       0.70      0.72      0.71        82
        109       0.94      0.82      0.88        78
        110       0.97      0.95      0.96       121
        111       0.66      0.74      0.70        54
        112       0.96      0.97      0.96        89
        113       0.73      0.97      0.83        31
        114       0.87      0.88      0.88       127
        115       0.96      0.94      0.95        78
        116       0.62      0.48      0.54       108
        117       0.86      0.87      0.86       110
        118       0.54      0.43      0.48        88
        119       0.90      0.77      0.83       107
        120       0.57      0.60      0.58        70
        121       0.77      0.95      0.85        39
        122       0.96      0.85      0.91       124
        123       0.88      0.76      0.82        59
        124       0.92      0.96      0.94        70
        125       0.74      0.90      0.81        39
        126       0.82      0.75      0.79       114
        127       1.00      1.00      1.00        56
        128       0.68      0.87      0.76        83
        129       0.97      0.91      0.94       125
        130       0.78      0.83      0.81        48
        131       0.85      0.88      0.87       112
        132       1.00      0.97      0.98        97
        133       0.99      0.75      0.85       164
        134       0.43      0.56      0.49        32
        135       0.82      0.88      0.85        81
        136       0.74      0.73      0.73       104
        137       0.71      0.71      0.71       134
        138       0.98      0.93      0.95        88
        139       0.33      0.13      0.19        38
        140       0.98      0.91      0.94        44
        141       0.87      0.84      0.85        62
        142       1.00      0.92      0.96       104
        143       0.63      0.73      0.67        51
        144       0.56      0.59      0.57        73
        145       0.70      0.86      0.77        50
        146       0.86      0.83      0.84        86
        147       0.88      0.87      0.88        93
        148       0.94      0.78      0.85        58
        149       0.93      0.90      0.91       110
        150       0.79      0.79      0.79        77
        151       0.66      0.63      0.64        75
        152       0.95      0.96      0.95        54
        153       0.48      0.94      0.64        32
        154       0.78      0.78      0.78        74
        155       0.89      0.73      0.80       189
        156       0.67      0.49      0.57        92
        157       0.88      0.94      0.91        31
        158       0.77      0.72      0.74       101
        159       0.55      0.57      0.56       109
        160       0.19      0.26      0.22        39
        161       0.68      0.75      0.72        57
        162       0.89      0.97      0.93        33
        163       0.40      0.40      0.40        70
        164       0.96      0.86      0.91       160
        165       0.37      0.40      0.39        40
        166       0.89      0.93      0.91        72
        167       0.76      0.84      0.80        64
        168       0.86      0.75      0.80        76
        169       0.75      0.82      0.79        40
        170       0.99      0.95      0.97       141
        171       0.18      0.06      0.10        62
        172       0.86      0.88      0.87        89
        173       0.65      0.72      0.68        39
        174       0.76      0.92      0.83        38
        175       0.80      0.96      0.87        45
        176       0.41      0.60      0.49        43
        177       0.70      0.88      0.78        32
        178       0.94      0.91      0.93        55
        179       0.44      0.56      0.49        32
        180       0.68      0.65      0.66        82
        181       0.58      0.59      0.58        32
        182       0.75      0.68      0.71        87
        183       0.67      0.86      0.75        42
        184       0.89      0.88      0.89        58
        185       0.72      0.87      0.79        30
        186       0.96      0.95      0.95        56
        187       0.68      0.97      0.80        31
        188       0.97      1.00      0.99        33
        189       0.77      0.91      0.83        47
        190       0.96      0.93      0.94       108
        191       0.98      1.00      0.99        40
        192       0.97      0.94      0.95       119
        193       0.97      1.00      0.99        33
        194       0.77      0.81      0.79       152
        195       0.95      0.83      0.89        71
        196       0.92      0.92      0.92        39
        197       0.95      0.94      0.95       124
        198       0.86      0.81      0.83        31
        199       0.48      0.55      0.51        38
        200       0.94      0.97      0.96        34
        201       0.32      0.39      0.35        31
        202       0.82      0.85      0.83        53
        203       0.89      0.91      0.90        34
        204       0.58      0.82      0.68        39
        205       0.64      0.72      0.67        75
        206       0.98      0.89      0.93       101
        207       0.33      0.36      0.34        45
        208       0.81      0.72      0.76       108
        209       0.90      0.94      0.92        80
        210       0.85      0.85      0.85        33
        211       0.86      0.95      0.90        60
        212       0.74      0.68      0.71        93
        213       0.82      0.77      0.79        87
        214       0.24      0.22      0.23        36
        215       0.48      0.49      0.49        92
        216       0.76      0.83      0.79        75
        217       0.55      0.58      0.56        71
        218       0.52      0.62      0.57        56
        219       0.87      0.78      0.82        60
        220       0.81      0.86      0.83        50
        221       0.56      0.65      0.60        43
        222       1.00      1.00      1.00        46
        223       0.75      0.71      0.73        97
        224       1.00      0.84      0.91        85
        225       0.44      0.73      0.55        33
        226       0.91      1.00      0.96        43
        227       0.96      0.94      0.95        54
        228       0.51      0.80      0.62        30
        229       0.35      0.69      0.46        32
        230       0.58      0.59      0.59        37
        231       0.92      0.83      0.88        59
        232       0.92      0.86      0.89        98
        233       0.96      0.93      0.94        70
        234       0.96      0.97      0.96        69
        235       0.92      0.86      0.89       104
        236       1.00      0.96      0.98        99
        237       0.92      0.87      0.90       123
        238       0.91      0.90      0.91        93
        239       1.00      0.96      0.98        48
        240       0.58      0.63      0.60        82
        241       0.76      0.85      0.80        33
        242       0.82      0.85      0.84        95
        243       0.90      0.89      0.90        63
        244       0.43      0.54      0.48        41
        245       0.78      0.70      0.74       104
        246       0.83      0.69      0.75        77
        247       0.95      1.00      0.98        62
        248       0.97      0.95      0.96        78
        249       0.85      0.75      0.80        61
        250       0.91      0.85      0.88       121
        251       0.63      0.55      0.59        31
        252       0.74      0.93      0.82        42
        253       0.61      0.67      0.64        45
        254       0.25      0.31      0.28        48
        255       1.00      1.00      1.00        42
        256       1.00      0.98      0.99        98
        257       0.88      0.94      0.91        32
        258       0.45      0.53      0.49        55
        259       0.92      0.96      0.94       128
        260       0.66      0.49      0.56       121
        261       0.59      0.85      0.69        39
        262       0.98      1.00      0.99        62
        263       0.49      0.67      0.57        64
        264       0.93      0.93      0.93        45
        265       0.86      0.96      0.91        56
        266       0.90      0.94      0.92        68
        267       0.87      0.87      0.87        67
        268       0.84      0.80      0.82       148
        269       0.64      0.81      0.71        31
        270       0.62      0.42      0.50       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.94      1.00      0.97        44
        274       0.79      0.77      0.78       168
        275       0.32      0.66      0.43        38
        276       0.25      0.51      0.33        35
        277       0.87      0.95      0.91        58
        278       0.68      0.76      0.71        33
        279       0.53      0.57      0.55        63
        280       0.95      0.88      0.91        67
        281       0.64      0.84      0.73        45
        282       0.79      0.94      0.86        49
        283       0.82      1.00      0.90        32
        284       0.72      0.61      0.66        96
        285       0.77      0.81      0.79       118
        286       0.85      0.86      0.86        66
        287       0.77      0.67      0.72       107
        288       0.94      0.94      0.94        35
        289       0.92      0.88      0.90        88
        290       0.83      0.80      0.82        66
        291       0.99      0.88      0.93        96
        292       0.99      0.83      0.90       135
        293       0.70      0.84      0.76        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.64      0.59      0.61        82
        297       0.90      0.84      0.87       107
        298       0.76      0.76      0.76        41
        299       0.86      0.81      0.83        31
        300       0.77      0.88      0.82        42
        301       0.73      0.83      0.78        53
        302       0.95      0.93      0.94        90
        303       0.56      0.77      0.65        35
        304       0.91      0.98      0.94        42
        305       0.94      0.89      0.91       102
        306       1.00      0.95      0.97        39
        307       0.89      0.77      0.83       120
        308       0.96      0.99      0.97       117
        309       0.91      0.86      0.88        98
        310       0.97      0.99      0.98       110
        311       0.44      0.47      0.45        36
        312       0.96      0.99      0.98        99
        313       0.78      0.94      0.85       116
        314       0.59      0.74      0.66        39
        315       0.60      0.70      0.65        30
        316       0.94      0.84      0.89       156
        317       0.76      0.84      0.80        75
        318       0.65      0.72      0.68        72
        319       0.98      0.90      0.94       154
        320       0.95      0.86      0.90        86
        321       0.27      0.17      0.21        59
        322       0.74      0.80      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.88      0.91        56
        325       0.57      0.57      0.57       112
        326       0.92      0.90      0.91       114
        327       0.86      0.85      0.85        93
        328       0.74      0.72      0.73        85
        329       0.33      0.17      0.22        76
        330       0.64      0.67      0.65        66
        331       0.86      0.90      0.88       128
        332       0.88      0.91      0.90        33
        333       0.49      0.55      0.52        56
        334       0.62      0.44      0.51        41
        335       0.87      0.65      0.74       103
        336       0.80      0.85      0.82        46
        337       0.96      0.93      0.94        54
        338       0.80      0.94      0.87        52
        339       0.72      0.68      0.70        57
        340       0.84      0.75      0.79       128
        341       0.91      0.93      0.92        42
        342       0.81      0.68      0.74       118
        343       0.51      0.58      0.54        45
        344       0.60      0.74      0.66        38
        345       0.99      0.97      0.98       153
        346       0.60      0.78      0.68        65
        347       0.91      0.83      0.87        64
        348       0.74      0.56      0.64        99
        349       1.00      0.99      1.00       105
        350       0.78      0.80      0.79        86
        351       0.95      0.96      0.96       125
        352       0.43      0.36      0.39        64
        353       0.42      0.56      0.48        50
        354       0.42      0.68      0.52        41
        355       0.93      0.87      0.90       108
        356       0.80      0.75      0.77        48
        357       0.71      0.66      0.68        98
        358       0.64      0.63      0.64        43
        359       0.73      0.75      0.74       120
        360       1.00      0.99      1.00       103
        361       0.43      0.44      0.44       120
        362       0.98      0.89      0.93       158
        363       0.89      0.98      0.93        51
        364       0.55      0.76      0.64        42
        365       0.89      0.85      0.87       122
        366       0.70      0.58      0.63       110
        367       1.00      0.94      0.97        67
        368       0.85      0.86      0.86       114
        369       0.68      0.84      0.75        62
        370       0.91      0.93      0.92       130
        371       0.67      0.71      0.69        62
        372       0.81      0.83      0.82       110
        373       0.89      0.89      0.89       145
        374       0.48      0.75      0.59        32
        375       0.47      0.36      0.41       100
        376       0.95      0.97      0.96        39
        377       0.68      0.74      0.71        38
        378       0.36      0.58      0.44        31
        379       0.79      0.81      0.80        54
        380       0.72      0.89      0.80        44
        381       0.34      0.38      0.36        48
        382       0.70      0.84      0.76        31
        383       0.74      0.79      0.77        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.74      0.80      0.77        61
        387       0.60      0.75      0.67        32
        388       0.68      0.67      0.68       115
        389       0.77      0.86      0.81        43
        390       0.99      0.97      0.98       158
        391       0.66      0.72      0.69        89
        392       0.84      0.96      0.90        50
        393       0.93      0.93      0.93        69
        394       0.83      0.87      0.85        55
        395       0.80      0.86      0.83        81
        396       0.73      0.75      0.74        59
        397       0.94      0.97      0.95       121
        398       0.88      0.94      0.91        80
        399       0.81      0.87      0.84        45
        400       0.56      0.53      0.54        38
        401       0.78      0.84      0.81       108
        402       0.91      0.91      0.91        34
        403       0.90      0.84      0.87       123
        404       0.85      0.91      0.88        55
        405       0.87      0.84      0.86       103
        406       0.80      0.87      0.83       111
        407       0.72      0.76      0.74        76
        408       0.64      0.90      0.75        30
        409       0.56      0.60      0.57        42
        410       0.78      0.44      0.57        63
        411       0.83      0.76      0.79       136
        412       0.59      0.52      0.55        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.38      0.38        88
        416       0.88      0.70      0.78       116
        417       0.94      0.94      0.94       124
        418       0.91      0.72      0.81       127
        419       0.84      0.89      0.87        76
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.65      0.79      0.71        42
        423       0.75      0.70      0.73       110
        424       0.69      0.97      0.81        30
        425       0.77      0.89      0.82        37
        426       0.69      0.78      0.74        32
        427       0.75      0.94      0.83        32
        428       0.48      0.67      0.56        55
        429       0.96      0.98      0.97        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.78      0.65      0.71        71
        433       0.47      0.51      0.49        55
        434       0.91      0.94      0.92        32
        435       0.93      0.82      0.87        66
        436       0.92      0.88      0.90        65
        437       0.75      0.94      0.83        32
        438       0.77      0.69      0.73        54
        439       0.79      0.73      0.76        30
        440       0.98      0.99      0.98        86
        441       0.39      0.64      0.48        33
        442       0.80      0.77      0.79       131
        443       0.54      0.64      0.59        86
        444       0.51      0.46      0.48        48
        445       0.76      0.97      0.85        38
        446       0.95      0.99      0.97       123
        447       0.59      0.71      0.65        63
        448       0.85      0.76      0.80        58
        449       0.84      0.95      0.89        57
        450       0.60      0.71      0.65        56
        451       0.68      0.78      0.73        51
        452       0.55      0.76      0.64        41
        453       0.97      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.79      0.87      0.83        63
        456       0.92      0.82      0.87        85
        457       0.90      0.78      0.84       137
        458       0.76      0.81      0.78        47
        459       0.62      0.41      0.50        87
        460       0.99      0.99      0.99        95
        461       0.50      0.76      0.60        46
        462       0.57      0.54      0.55        69
        463       0.89      0.72      0.79       120
        464       0.87      0.83      0.85        93
        465       0.75      0.90      0.82        63
        466       0.42      0.56      0.48        32
        467       0.77      0.70      0.73        73
        468       0.93      0.98      0.95        42
        469       0.92      0.87      0.89        78
        470       1.00      0.97      0.99        36
        471       0.81      0.66      0.73        44
        472       0.67      0.66      0.67       112
        473       0.83      0.94      0.88        36
        474       0.54      0.47      0.50        79
        475       0.47      0.74      0.58        31
        476       0.99      0.90      0.94        78
        477       0.93      0.83      0.88       107
        478       0.90      0.58      0.70        33
        479       0.88      0.94      0.91        63
        480       0.91      0.95      0.93        91
        481       0.71      0.76      0.74        46
        482       0.91      0.88      0.89        33
        483       0.99      0.86      0.92       167
        484       1.00      0.97      0.99        35
        485       0.84      0.75      0.79        36
        486       0.30      0.31      0.30        74
        487       0.76      0.78      0.77        60
        488       0.57      0.68      0.62        31
        489       0.92      0.96      0.94        49
        490       0.82      0.66      0.73       121
        491       0.82      0.85      0.84        39
        492       0.75      0.75      0.75       114
        493       0.93      0.85      0.89       149
        494       0.64      0.75      0.69        51
        495       0.91      0.93      0.92        80
        496       0.81      0.97      0.88        39
        497       0.63      0.48      0.54        88

avg / total       0.80      0.80      0.80     35746

Iteration number, batch number :  8 0
Training data accuracy :  0.846385542169
Training data loss     :  0.00508250594895
Iteration number, batch number :  8 1
Training data accuracy :  0.829317269076
Training data loss     :  0.00516974095309
Iteration number, batch number :  8 2
Training data accuracy :  0.832329317269
Training data loss     :  0.00514533189866
Iteration number, batch number :  8 3
Training data accuracy :  0.843373493976
Training data loss     :  0.005145733717
Iteration number, batch number :  8 4
Training data accuracy :  0.840361445783
Training data loss     :  0.00512701473841
Iteration number, batch number :  8 5
Training data accuracy :  0.85843373494
Training data loss     :  0.00492103331625
Iteration number, batch number :  8 6
Training data accuracy :  0.845381526104
Training data loss     :  0.00529911742519
Iteration number, batch number :  8 7
Training data accuracy :  0.859437751004
Training data loss     :  0.00449390269454
Iteration number, batch number :  8 8
Training data accuracy :  0.85140562249
Training data loss     :  0.00492235138124
Iteration number, batch number :  8 9
Training data accuracy :  0.850401606426
Training data loss     :  0.00500095771029
Iteration number, batch number :  8 10
Training data accuracy :  0.868473895582
Training data loss     :  0.00443043682804
Iteration number, batch number :  8 11
Training data accuracy :  0.846385542169
Training data loss     :  0.00482956473295
Iteration number, batch number :  8 12
Training data accuracy :  0.86546184739
Training data loss     :  0.0047514450425
Iteration number, batch number :  8 13
Training data accuracy :  0.847389558233
Training data loss     :  0.0052434482868
Iteration number, batch number :  8 14
Training data accuracy :  0.861445783133
Training data loss     :  0.00457850416956
Iteration number, batch number :  8 15
Training data accuracy :  0.861445783133
Training data loss     :  0.00449990172292
Iteration number, batch number :  8 16
Training data accuracy :  0.879518072289
Training data loss     :  0.00422628080732
Iteration number, batch number :  8 17
Training data accuracy :  0.860441767068
Training data loss     :  0.0044993039618
Iteration number, batch number :  8 18
Training data accuracy :  0.853413654618
Training data loss     :  0.00488882229758
Iteration number, batch number :  8 19
Training data accuracy :  0.86546184739
Training data loss     :  0.00471651197454
Iteration number, batch number :  8 20
Training data accuracy :  0.869477911647
Training data loss     :  0.00445749556776
Iteration number, batch number :  8 21
Training data accuracy :  0.846385542169
Training data loss     :  0.00452823559411
Iteration number, batch number :  8 22
Training data accuracy :  0.86546184739
Training data loss     :  0.00429191151703
Iteration number, batch number :  8 23
Training data accuracy :  0.859437751004
Training data loss     :  0.00425884247947
Iteration number, batch number :  8 24
Training data accuracy :  0.850401606426
Training data loss     :  0.00474664247647
Iteration number, batch number :  8 25
Training data accuracy :  0.853413654618
Training data loss     :  0.00482361752674
Iteration number, batch number :  8 26
Training data accuracy :  0.871485943775
Training data loss     :  0.00453736849471
Iteration number, batch number :  8 27
Training data accuracy :  0.85843373494
Training data loss     :  0.00460654702011
Iteration number, batch number :  8 28
Training data accuracy :  0.876506024096
Training data loss     :  0.00418684295949
Iteration number, batch number :  8 29
Training data accuracy :  0.857429718876
Training data loss     :  0.00465904066496
Iteration number, batch number :  8 30
Training data accuracy :  0.853413654618
Training data loss     :  0.00449862541228
Iteration number, batch number :  8 31
Training data accuracy :  0.849397590361
Training data loss     :  0.00476217078587
Iteration number, batch number :  8 32
Training data accuracy :  0.85843373494
Training data loss     :  0.00462292018472
Iteration number, batch number :  8 33
Training data accuracy :  0.889558232932
Training data loss     :  0.00425962093196
Iteration number, batch number :  8 34
Training data accuracy :  0.880522088353
Training data loss     :  0.00413172336101
Iteration number, batch number :  8 35
Training data accuracy :  0.877510040161
Training data loss     :  0.00398202351954
Iteration number, batch number :  8 36
Training data accuracy :  0.86546184739
Training data loss     :  0.00423253525908
Iteration number, batch number :  8 37
Training data accuracy :  0.882530120482
Training data loss     :  0.00430228186118
Iteration number, batch number :  8 38
Training data accuracy :  0.875502008032
Training data loss     :  0.00433537519508
Iteration number, batch number :  8 39
Training data accuracy :  0.893574297189
Training data loss     :  0.0038220315478
Iteration number, batch number :  8 40
Training data accuracy :  0.860441767068
Training data loss     :  0.00468947292467
Iteration number, batch number :  8 41
Training data accuracy :  0.86546184739
Training data loss     :  0.00428006469121
Iteration number, batch number :  8 42
Training data accuracy :  0.879518072289
Training data loss     :  0.00425464359513
Iteration number, batch number :  8 43
Training data accuracy :  0.874497991968
Training data loss     :  0.00427528117837
Iteration number, batch number :  8 44
Training data accuracy :  0.871485943775
Training data loss     :  0.00387259773298
Iteration number, batch number :  8 45
Training data accuracy :  0.879518072289
Training data loss     :  0.0041289419936
Iteration number, batch number :  8 46
Training data accuracy :  0.872489959839
Training data loss     :  0.00406193941466
Iteration number, batch number :  8 47
Training data accuracy :  0.879518072289
Training data loss     :  0.00439805625384
Iteration number, batch number :  8 48
Training data accuracy :  0.860441767068
Training data loss     :  0.00454745643691
Iteration number, batch number :  8 49
Training data accuracy :  0.859437751004
Training data loss     :  0.00424637782498
Iteration number, batch number :  8 50
Training data accuracy :  0.871485943775
Training data loss     :  0.00428548660078
Iteration number, batch number :  8 51
Training data accuracy :  0.847389558233
Training data loss     :  0.00461317416198
Iteration number, batch number :  8 52
Training data accuracy :  0.857429718876
Training data loss     :  0.00510626020216
Iteration number, batch number :  8 53
2017-07-01 16:25:05.486330: I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 27891474 get requests, put_count=27891392 evicted_count=19000 eviction_rate=0.000681214 and unsatisfied allocation rate=0.00071477
Training data accuracy :  0.85140562249
Training data loss     :  0.00480138327563
Iteration number, batch number :  8 54
Training data accuracy :  0.860441767068
Training data loss     :  0.00479581410298
Iteration number, batch number :  8 55
Training data accuracy :  0.852409638554
Training data loss     :  0.00465319403366
Iteration number, batch number :  8 56
Training data accuracy :  0.859437751004
Training data loss     :  0.00487564085183
Iteration number, batch number :  8 57
Training data accuracy :  0.84437751004
Training data loss     :  0.00497930162986
Iteration number, batch number :  8 58
Training data accuracy :  0.841365461847
Training data loss     :  0.00495868266356
Iteration number, batch number :  8 59
Training data accuracy :  0.849397590361
Training data loss     :  0.00520323837341
Iteration number, batch number :  8 60
Training data accuracy :  0.854417670683
Training data loss     :  0.00488481728121
Iteration number, batch number :  8 61
Training data accuracy :  0.84437751004
Training data loss     :  0.0050863021271
Iteration number, batch number :  8 62
Training data accuracy :  0.84437751004
Training data loss     :  0.00500729548475
Iteration number, batch number :  8 63
Training data accuracy :  0.845381526104
Training data loss     :  0.00512601225392
Iteration number, batch number :  8 64
Training data accuracy :  0.852409638554
Training data loss     :  0.00539015085372
Iteration number, batch number :  8 65
Training data accuracy :  0.840361445783
Training data loss     :  0.00556359978602
Iteration number, batch number :  8 66
Training data accuracy :  0.847389558233
Training data loss     :  0.00487430539921
Iteration number, batch number :  8 67
Training data accuracy :  0.856425702811
Training data loss     :  0.00505888355357
Iteration number, batch number :  8 68
Training data accuracy :  0.849397590361
Training data loss     :  0.00528711945854
Iteration number, batch number :  8 69
Training data accuracy :  0.843373493976
Training data loss     :  0.00499915739732
Iteration number, batch number :  8 35
Iteration number, batch number :  8 34
Iteration number, batch number :  8 33
Iteration number, batch number :  8 32
Iteration number, batch number :  8 31
Iteration number, batch number :  8 30
Iteration number, batch number :  8 29
Iteration number, batch number :  8 28
Iteration number, batch number :  8 27
Iteration number, batch number :  8 26
Iteration number, batch number :  8 25
Iteration number, batch number :  8 24
Iteration number, batch number :  8 23
Iteration number, batch number :  8 22
Iteration number, batch number :  8 21
Iteration number, batch number :  8 20
Iteration number, batch number :  8 19
Iteration number, batch number :  8 18
Iteration number, batch number :  8 17
Iteration number, batch number :  8 16
Iteration number, batch number :  8 15
Iteration number, batch number :  8 14
Iteration number, batch number :  8 13
Iteration number, batch number :  8 12
Iteration number, batch number :  8 11
Iteration number, batch number :  8 10
Iteration number, batch number :  8 9
Iteration number, batch number :  8 8
Iteration number, batch number :  8 7
Iteration number, batch number :  8 6
Iteration number, batch number :  8 5
Iteration number, batch number :  8 4
Iteration number, batch number :  8 3
Iteration number, batch number :  8 2
Iteration number, batch number :  8 1
Iteration number, batch number :  8 0
Accuracy on test data :  76.7017919531 27223.0 27223
Lengths of y_actual and y_predicted :  35492 35492
             precision    recall  f1-score   support

          0       0.79      0.91      0.85        45
          1       0.74      0.95      0.83        41
          2       0.64      0.81      0.71        37
          3       0.83      0.96      0.89        52
          4       0.94      0.55      0.70       130
          5       0.92      0.97      0.95        36
          6       0.50      0.71      0.59        51
          7       0.71      0.62      0.66        47
          8       0.39      0.38      0.38        76
          9       0.96      1.00      0.98        51
         10       0.95      0.93      0.94        45
         11       0.37      0.45      0.41        31
         12       0.93      0.86      0.89        79
         13       0.71      0.82      0.77        85
         14       0.93      0.87      0.90       143
         15       0.51      0.42      0.46       120
         16       0.91      0.94      0.93        34
         17       0.65      0.31      0.42       124
         18       0.77      0.81      0.79        70
         19       0.54      0.55      0.55        91
         20       0.41      0.62      0.49        39
         21       0.90      0.34      0.49        53
         22       0.53      0.51      0.52        45
         23       0.48      0.73      0.58        33
         24       0.79      0.78      0.78        40
         25       0.97      0.98      0.97        85
         26       0.93      0.73      0.82       132
         27       0.93      0.90      0.92        90
         28       0.37      0.53      0.44        30
         29       0.84      0.84      0.84        96
         30       0.90      0.86      0.88        74
         31       0.96      0.89      0.92        79
         32       1.00      0.99      0.99       269
         33       0.97      0.94      0.95        31
         34       0.40      0.66      0.50        32
         35       0.85      0.95      0.90        77
         36       0.87      0.92      0.90        66
         37       0.75      0.85      0.80       114
         38       0.04      0.01      0.01       121
         39       0.76      0.85      0.81        34
         40       0.33      0.45      0.38        76
         41       0.91      0.65      0.76        92
         42       0.65      0.74      0.69        58
         43       0.93      0.87      0.90        63
         44       0.57      0.80      0.67        30
         45       0.81      0.95      0.87        40
         46       0.00      0.00      0.00       161
         47       0.58      0.79      0.67        47
         48       0.89      0.94      0.92        36
         49       0.83      0.95      0.89        37
         50       0.81      0.65      0.72       123
         51       0.62      0.47      0.53       101
         52       0.96      0.94      0.95        51
         53       1.00      0.92      0.96        76
         54       0.94      0.90      0.92        70
         55       0.99      0.94      0.96        94
         56       0.59      0.65      0.62        66
         57       0.70      0.67      0.68        79
         58       0.97      0.92      0.94        98
         59       0.95      0.90      0.93        92
         60       1.00      0.88      0.94        33
         61       0.86      0.93      0.89        41
         62       0.33      0.07      0.11        30
         63       0.84      0.88      0.86        74
         64       0.88      1.00      0.94        38
         65       0.95      0.88      0.92        95
         66       0.80      0.70      0.75        94
         67       0.88      0.92      0.90       132
         68       0.37      0.43      0.40        72
         69       0.34      0.50      0.41        30
         70       0.74      0.81      0.77        42
         71       0.67      0.82      0.73        44
         72       0.36      0.67      0.47        42
         73       0.82      0.84      0.83        37
         74       0.64      0.89      0.74        98
         75       0.95      0.88      0.91       163
         76       0.91      0.91      0.91        65
         77       0.97      0.94      0.96        70
         78       0.82      0.85      0.84        33
         79       0.59      0.82      0.68        33
         80       0.77      0.66      0.71        71
         81       0.96      0.96      0.96       131
         82       0.85      0.68      0.76       174
         83       0.71      0.74      0.73        47
         84       0.22      0.27      0.24        44
         85       0.96      0.93      0.94       110
         86       0.55      0.45      0.50        97
         87       0.92      0.78      0.84        83
         88       0.75      0.84      0.79        43
         89       0.73      0.91      0.81        35
         90       0.70      0.82      0.76        38
         91       0.70      0.61      0.65        51
         92       0.43      0.72      0.54        40
         93       0.94      0.92      0.93        48
         94       0.90      0.90      0.90        71
         95       0.85      0.79      0.82       117
         96       0.55      0.84      0.67        38
         97       0.69      0.90      0.78        30
         98       0.88      0.85      0.86        98
         99       0.61      0.75      0.68        69
        100       0.92      0.91      0.91        98
        101       0.66      0.65      0.65        85
        102       0.41      0.57      0.47        56
        103       0.97      0.92      0.94        36
        104       0.69      0.60      0.64        40
        105       0.69      0.55      0.62        65
        106       0.63      0.75      0.69        32
        107       1.00      0.56      0.72        34
        108       0.70      0.75      0.73        81
        109       0.57      0.22      0.32        77
        110       0.96      0.91      0.94       121
        111       0.58      0.57      0.57        53
        112       0.92      0.89      0.90        88
        113       0.78      0.97      0.87        30
        114       0.84      0.88      0.86       127
        115       0.92      0.73      0.81        78
        116       0.59      0.53      0.56       108
        117       0.91      0.83      0.87       110
        118       0.52      0.42      0.47        88
        119       0.91      0.81      0.86       106
        120       0.63      0.75      0.68        69
        121       0.79      0.85      0.81        39
        122       0.89      0.77      0.83       124
        123       0.86      0.83      0.84        59
        124       1.00      0.96      0.98        69
        125       0.75      0.87      0.80        38
        126       0.77      0.66      0.71       113
        127       0.98      1.00      0.99        55
        128       0.71      0.85      0.77        82
        129       0.93      0.89      0.91       125
        130       0.59      0.85      0.70        47
        131       0.83      0.35      0.49       112
        132       0.99      1.00      0.99        96
        133       0.98      0.74      0.84       164
        134       0.47      0.62      0.53        32
        135       0.80      0.81      0.81        80
        136       0.74      0.71      0.72       103
        137       0.76      0.82      0.79       134
        138       0.92      0.91      0.91        88
        139       0.17      0.08      0.11        37
        140       0.92      1.00      0.96        44
        141       0.82      0.92      0.87        61
        142       0.97      0.94      0.96       103
        143       0.55      0.69      0.61        51
        144       0.48      0.65      0.55        72
        145       0.72      0.96      0.82        49
        146       0.77      0.76      0.77        85
        147       0.91      0.87      0.89        93
        148       0.86      0.86      0.86        57
        149       0.85      0.86      0.85       109
        150       0.86      0.64      0.74        76
        151       0.65      0.67      0.66        75
        152       0.94      0.91      0.92        54
        153       0.45      0.78      0.57        32
        154       0.90      0.88      0.89        73
        155       0.85      0.61      0.71       189
        156       0.67      0.50      0.57        92
        157       0.94      0.97      0.95        31
        158       0.47      0.15      0.23       100
        159       0.52      0.43      0.47       108
        160       0.26      0.44      0.32        39
        161       0.69      0.74      0.71        57
        162       0.92      1.00      0.96        33
        163       0.39      0.47      0.43        70
        164       0.97      0.81      0.88       159
        165       0.48      0.72      0.58        39
        166       0.94      0.94      0.94        71
        167       0.86      0.80      0.83        64
        168       0.85      0.81      0.83        75
        169       0.52      0.64      0.57        39
        170       0.99      0.95      0.97       140
        171       0.19      0.10      0.13        61
        172       0.78      0.89      0.83        88
        173       0.63      0.69      0.66        39
        174       0.82      0.89      0.86        37
        175       0.77      0.82      0.80        45
        176       0.36      0.50      0.42        42
        177       0.63      0.87      0.73        31
        178       0.95      0.95      0.95        55
        179       0.36      0.58      0.44        31
        180       0.66      0.62      0.64        82
        181       0.57      0.65      0.61        31
        182       0.72      0.68      0.70        87
        183       0.76      0.90      0.82        41
        184       0.88      0.91      0.90        57
        185       0.74      0.87      0.80        30
        186       1.00      0.95      0.97        56
        187       0.78      0.97      0.87        30
        188       0.97      1.00      0.99        33
        189       0.69      0.78      0.73        46
        190       0.91      0.99      0.95       107
        191       0.93      0.95      0.94        40
        192       0.86      0.73      0.79       118
        193       0.94      0.97      0.96        33
        194       0.81      0.74      0.77       152
        195       0.91      0.85      0.88        71
        196       0.97      0.95      0.96        38
        197       1.00      0.86      0.93       124
        198       0.91      0.94      0.92        31
        199       0.40      0.70      0.51        37
        200       0.97      1.00      0.99        34
        201       0.16      0.27      0.20        30
        202       0.70      0.81      0.75        53
        203       0.86      0.97      0.91        33
        204       0.69      0.92      0.79        38
        205       0.53      0.56      0.55        75
        206       0.95      0.93      0.94       101
        207       0.27      0.27      0.27        45
        208       0.79      0.76      0.77       108
        209       0.78      0.85      0.81        79
        210       0.73      0.94      0.82        32
        211       0.90      0.90      0.90        60
        212       0.74      0.72      0.73        93
        213       0.77      0.79      0.78        87
        214       0.23      0.34      0.28        35
        215       0.56      0.56      0.56        91
        216       0.79      0.56      0.66        75
        217       0.54      0.61      0.58        70
        218       0.55      0.69      0.61        55
        219       0.92      0.73      0.81        60
        220       0.84      0.92      0.88        50
        221       0.51      0.69      0.59        42
        222       1.00      0.96      0.98        45
        223       0.80      0.74      0.77        96
        224       0.91      0.84      0.87        85
        225       0.55      0.84      0.67        32
        226       0.95      0.98      0.97        43
        227       0.90      0.98      0.94        54
        228       0.43      0.87      0.58        30
        229       0.25      0.50      0.34        32
        230       0.52      0.62      0.57        37
        231       0.80      0.90      0.85        59
        232       0.96      0.82      0.89        97
        233       0.97      0.94      0.96        69
        234       0.93      0.55      0.69        69
        235       0.86      0.76      0.80       103
        236       0.99      0.98      0.98        98
        237       0.89      0.71      0.79       123
        238       0.87      0.90      0.88        93
        239       1.00      0.96      0.98        48
        240       0.63      0.65      0.64        81
        241       0.86      0.94      0.90        32
        242       0.88      0.75      0.81        95
        243       0.81      0.87      0.84        63
        244       0.45      0.60      0.52        40
        245       0.78      0.77      0.77       103
        246       0.71      0.52      0.60        77
        247       0.94      1.00      0.97        61
        248       0.95      0.90      0.92        78
        249       0.91      0.83      0.87        60
        250       0.94      0.93      0.94       120
        251       0.56      0.73      0.64        30
        252       0.82      0.86      0.84        42
        253       0.56      0.64      0.60        44
        254       0.26      0.34      0.29        47
        255       1.00      0.95      0.97        41
        256       1.00      0.94      0.97        98
        257       0.78      0.94      0.85        31
        258       0.36      0.33      0.34        55
        259       0.97      0.92      0.94       128
        260       0.62      0.45      0.53       121
        261       0.65      0.85      0.73        39
        262       0.97      0.98      0.98        62
        263       0.50      0.70      0.58        63
        264       0.87      0.89      0.88        45
        265       0.72      0.91      0.81        55
        266       0.91      0.94      0.93        67
        267       0.86      0.77      0.82        66
        268       0.80      0.70      0.75       148
        269       0.55      0.90      0.68        30
        270       0.59      0.47      0.52       149
        271       0.78      0.90      0.84        42
        272       0.97      0.95      0.96       119
        273       0.91      1.00      0.96        43
        274       0.80      0.75      0.78       167
        275       0.34      0.54      0.42        37
        276       0.20      0.40      0.26        35
        277       0.93      0.96      0.95        57
        278       0.41      0.28      0.33        32
        279       0.48      0.53      0.50        62
        280       0.00      0.00      0.00        66
        281       0.68      0.89      0.77        45
        282       0.84      0.90      0.87        48
        283       0.79      0.97      0.87        31
        284       0.72      0.67      0.69        96
        285       0.79      0.75      0.77       117
        286       0.97      0.85      0.90        66
        287       0.74      0.73      0.73       107
        288       0.70      0.89      0.78        35
        289       0.95      0.92      0.94        87
        290       0.73      0.73      0.73        66
        291       0.96      0.89      0.92        96
        292       0.97      0.70      0.82       135
        293       0.62      0.91      0.74        43
        294       0.47      0.87      0.61        31
        295       0.77      0.86      0.81        56
        296       0.48      0.53      0.51        81
        297       0.89      0.89      0.89       107
        298       0.70      0.68      0.69        41
        299       0.67      0.90      0.77        31
        300       0.82      0.80      0.81        41
        301       0.69      0.81      0.74        52
        302       0.80      0.91      0.85        90
        303       0.49      0.82      0.62        34
        304       0.91      0.95      0.93        41
        305       0.96      0.87      0.91       101
        306       0.95      0.95      0.95        38
        307       0.90      0.67      0.77       120
        308       0.97      0.96      0.96       117
        309       0.90      0.87      0.88        97
        310       0.95      0.95      0.95       109
        311       0.40      0.54      0.46        35
        312       0.95      0.90      0.92        98
        313       0.67      0.92      0.78       116
        314       0.47      0.72      0.57        39
        315       0.71      0.83      0.77        30
        316       0.95      0.90      0.92       155
        317       0.75      0.85      0.80        75
        318       0.71      0.65      0.68        72
        319       0.98      0.86      0.91       154
        320       0.85      0.84      0.84        85
        321       0.07      0.03      0.05        58
        322       0.78      0.82      0.80       119
        323       1.00      1.00      1.00        43
        324       0.95      0.96      0.95        55
        325       0.60      0.59      0.59       112
        326       0.91      0.88      0.89       114
        327       0.86      0.82      0.84        92
        328       0.71      0.71      0.71        84
        329       0.31      0.16      0.21        75
        330       0.68      0.68      0.68        66
        331       0.93      0.87      0.90       127
        332       0.84      0.94      0.89        33
        333       0.50      0.58      0.54        55
        334       0.50      0.29      0.37        41
        335       0.82      0.79      0.80       103
        336       0.74      0.89      0.81        45
        337       0.89      0.93      0.91        54
        338       0.79      0.88      0.83        51
        339       0.70      0.77      0.73        57
        340       0.79      0.76      0.77       127
        341       0.93      0.95      0.94        41
        342       0.75      0.71      0.73       118
        343       0.59      0.67      0.62        45
        344       0.61      0.89      0.73        37
        345       0.99      0.97      0.98       152
        346       0.58      0.69      0.63        64
        347       0.89      0.90      0.90        63
        348       0.75      0.58      0.66        98
        349       0.99      0.98      0.99       104
        350       0.77      0.85      0.81        86
        351       0.89      0.87      0.88       124
        352       0.46      0.60      0.52        63
        353       0.54      0.52      0.53        50
        354       0.49      0.70      0.58        40
        355       0.95      0.87      0.91       108
        356       0.82      0.89      0.86        47
        357       0.71      0.58      0.64        98
        358       0.65      0.81      0.72        42
        359       0.74      0.84      0.79       120
        360       1.00      0.94      0.97       102
        361       0.50      0.44      0.47       120
        362       0.94      0.93      0.94       157
        363       0.91      0.98      0.94        51
        364       0.46      0.67      0.54        42
        365       0.94      0.89      0.91       122
        366       0.69      0.61      0.65       109
        367       1.00      0.97      0.98        67
        368       0.83      0.86      0.84       113
        369       0.55      0.69      0.61        62
        370       0.97      0.92      0.94       130
        371       0.57      0.72      0.64        61
        372       0.81      0.84      0.83       109
        373       0.83      0.80      0.82       145
        374       0.48      0.71      0.57        31
        375       0.53      0.36      0.43        99
        376       0.88      0.92      0.90        38
        377       0.74      0.78      0.76        37
        378       0.25      0.65      0.36        31
        379       0.76      0.83      0.80        54
        380       0.70      0.91      0.79        44
        381       0.23      0.38      0.28        48
        382       0.71      0.94      0.81        31
        383       0.69      0.75      0.72        89
        384       0.99      0.99      0.99       100
        385       0.98      0.89      0.93        45
        386       0.82      0.83      0.83        60
        387       0.62      0.77      0.69        31
        388       0.67      0.74      0.71       115
        389       0.76      0.86      0.80        43
        390       0.99      0.99      0.99       158
        391       0.68      0.57      0.62        89
        392       0.85      0.94      0.90        50
        393       0.98      0.91      0.95        69
        394       0.78      0.93      0.85        55
        395       0.73      0.81      0.77        80
        396       0.63      0.78      0.70        58
        397       0.96      0.93      0.94       121
        398       0.91      0.94      0.93        80
        399       0.95      0.93      0.94        44
        400       0.45      0.68      0.54        37
        401       0.86      0.76      0.81       108
        402       0.94      0.88      0.91        33
        403       0.93      0.88      0.90       123
        404       0.84      0.95      0.89        55
        405       0.90      0.72      0.80       102
        406       0.82      0.83      0.82       110
        407       0.61      0.62      0.61        76
        408       0.65      0.93      0.77        30
        409       0.49      0.60      0.54        42
        410       0.79      0.42      0.55        62
        411       0.74      0.71      0.72       135
        412       0.50      0.53      0.52        64
        413       0.97      0.97      0.97        31
        414       0.95      0.92      0.94        66
        415       0.30      0.27      0.28        88
        416       0.84      0.82      0.83       115
        417       0.97      0.94      0.96       124
        418       0.91      0.78      0.84       126
        419       0.80      0.47      0.59        75
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.52      0.81      0.63        42
        423       0.66      0.67      0.67       109
        424       0.91      1.00      0.95        30
        425       0.76      0.95      0.84        37
        426       0.45      0.65      0.53        31
        427       0.70      0.97      0.82        32
        428       0.53      0.70      0.60        54
        429       0.98      0.98      0.98        87
        430       0.96      0.95      0.95        56
        431       0.98      0.98      0.98        42
        432       0.70      0.69      0.69        70
        433       0.39      0.49      0.43        55
        434       0.97      1.00      0.98        32
        435       0.97      0.94      0.95        65
        436       0.54      0.78      0.64        64
        437       0.70      0.88      0.78        32
        438       0.82      0.68      0.74        53
        439       0.66      0.70      0.68        30
        440       0.96      0.96      0.96        85
        441       0.36      0.64      0.46        33
        442       0.78      0.82      0.80       131
        443       0.54      0.61      0.57        85
        444       0.55      0.70      0.62        47
        445       0.82      0.87      0.85        38
        446       0.96      0.86      0.91       122
        447       0.61      0.49      0.54        63
        448       0.90      0.91      0.90        57
        449       0.88      1.00      0.93        57
        450       0.56      0.73      0.64        56
        451       0.62      0.71      0.66        51
        452       0.48      0.53      0.50        40
        453       0.95      0.91      0.93       137
        454       0.97      0.92      0.95        39
        455       0.73      0.87      0.80        63
        456       0.92      0.93      0.92        84
        457       0.91      0.78      0.84       136
        458       0.58      0.96      0.72        47
        459       0.58      0.44      0.50        87
        460       0.95      0.96      0.95        94
        461       0.46      0.80      0.58        46
        462       0.51      0.50      0.50        68
        463       0.79      0.81      0.80       120
        464       0.90      0.85      0.87        93
        465       0.63      0.84      0.72        62
        466       0.47      0.71      0.56        31
        467       0.75      0.69      0.72        72
        468       0.89      0.98      0.93        41
        469       0.97      0.88      0.93        78
        470       0.97      0.92      0.94        36
        471       0.40      0.05      0.08        44
        472       0.62      0.71      0.66       112
        473       0.92      0.94      0.93        35
        474       0.49      0.50      0.49        78
        475       0.58      0.83      0.68        30
        476       0.91      0.86      0.88        78
        477       0.88      0.79      0.83       106
        478       0.75      0.47      0.58        32
        479       0.78      0.90      0.84        63
        480       0.94      0.98      0.96        91
        481       0.79      0.84      0.82        45
        482       0.91      0.91      0.91        32
        483       0.99      0.92      0.95       166
        484       0.94      1.00      0.97        34
        485       0.62      0.89      0.73        35
        486       0.33      0.33      0.33        73
        487       0.62      0.64      0.63        59
        488       0.50      0.87      0.64        31
        489       0.79      0.94      0.86        48
        490       0.62      0.70      0.66       121
        491       0.00      0.00      0.00        39
        492       0.83      0.78      0.81       114
        493       0.96      0.89      0.92       149
        494       0.68      0.71      0.69        51
        495       0.90      0.88      0.89        80
        496       0.93      1.00      0.96        38
        497       0.48      0.53      0.51        88

avg / total       0.77      0.77      0.76     35492

Iteration number, batch number :  8 35
Iteration number, batch number :  8 34
Iteration number, batch number :  8 33
Iteration number, batch number :  8 32
Iteration number, batch number :  8 31
Iteration number, batch number :  8 30
Iteration number, batch number :  8 29
Iteration number, batch number :  8 28
Iteration number, batch number :  8 27
Iteration number, batch number :  8 26
Iteration number, batch number :  8 25
Iteration number, batch number :  8 24
Iteration number, batch number :  8 23
Iteration number, batch number :  8 22
Iteration number, batch number :  8 21
Iteration number, batch number :  8 20
Iteration number, batch number :  8 19
Iteration number, batch number :  8 18
Iteration number, batch number :  8 17
Iteration number, batch number :  8 16
Iteration number, batch number :  8 15
Iteration number, batch number :  8 14
Iteration number, batch number :  8 13
Iteration number, batch number :  8 12
Iteration number, batch number :  8 11
Iteration number, batch number :  8 10
Iteration number, batch number :  8 9
Iteration number, batch number :  8 8
Iteration number, batch number :  8 7
Iteration number, batch number :  8 6
Iteration number, batch number :  8 5
Iteration number, batch number :  8 4
Iteration number, batch number :  8 3
Iteration number, batch number :  8 2
Iteration number, batch number :  8 1
Iteration number, batch number :  8 0
Accuracy on cv data :  79.72920047 28500.0 28500
Lengths of y_actual and y_predicted :  35746 35746
             precision    recall  f1-score   support

          0       0.83      0.93      0.88        46
          1       0.75      0.90      0.82        42
          2       0.67      0.92      0.77        37
          3       0.95      1.00      0.97        53
          4       0.97      0.85      0.91       131
          5       0.87      0.92      0.89        37
          6       0.48      0.71      0.57        51
          7       0.68      0.57      0.62        47
          8       0.62      0.57      0.59        77
          9       0.96      1.00      0.98        51
         10       0.93      0.93      0.93        46
         11       0.76      0.91      0.83        32
         12       0.87      0.94      0.90        79
         13       0.77      0.89      0.83        85
         14       0.93      0.97      0.95       143
         15       0.48      0.39      0.43       121
         16       0.85      0.94      0.89        35
         17       0.84      0.74      0.79       125
         18       0.78      0.83      0.81        70
         19       0.65      0.76      0.70        92
         20       0.68      0.77      0.72        39
         21       0.96      0.94      0.95        53
         22       0.73      0.72      0.73        46
         23       0.59      0.91      0.71        33
         24       0.77      0.90      0.83        40
         25       0.99      0.98      0.98        86
         26       0.96      0.90      0.93       132
         27       0.81      0.87      0.84        90
         28       0.55      0.84      0.67        31
         29       0.88      0.88      0.88        96
         30       0.89      0.89      0.89        74
         31       0.97      0.97      0.97        80
         32       1.00      0.98      0.99       269
         33       0.97      1.00      0.98        32
         34       0.61      0.78      0.68        32
         35       0.86      0.92      0.89        77
         36       0.90      0.95      0.93        66
         37       0.75      0.77      0.76       115
         38       0.14      0.04      0.06       121
         39       0.86      0.91      0.89        35
         40       0.35      0.38      0.36        76
         41       0.89      0.83      0.86        93
         42       0.68      0.83      0.75        59
         43       0.92      0.95      0.94        63
         44       0.68      0.84      0.75        31
         45       0.86      0.93      0.89        40
         46       0.86      0.60      0.71       161
         47       0.61      0.77      0.68        48
         48       0.91      0.84      0.87        37
         49       0.76      0.89      0.82        38
         50       0.83      0.73      0.77       124
         51       0.56      0.42      0.48       101
         52       0.96      0.96      0.96        51
         53       0.97      0.99      0.98        77
         54       0.94      0.93      0.94        71
         55       1.00      0.96      0.98        95
         56       0.59      0.53      0.56        66
         57       0.79      0.75      0.77        80
         58       0.98      0.99      0.98        99
         59       0.97      0.90      0.93        93
         60       0.97      0.94      0.96        34
         61       0.85      1.00      0.92        41
         62       0.79      0.84      0.81        31
         63       0.99      0.92      0.95        74
         64       0.80      0.97      0.88        38
         65       0.90      0.88      0.89        96
         66       0.83      0.87      0.85        95
         67       0.88      0.92      0.90       132
         68       0.56      0.56      0.56        73
         69       0.45      0.43      0.44        30
         70       0.79      0.88      0.84        43
         71       0.69      0.66      0.67        44
         72       0.44      0.62      0.51        42
         73       0.91      0.84      0.87        37
         74       0.71      0.86      0.78        99
         75       0.90      0.87      0.88       164
         76       0.95      0.95      0.95        65
         77       0.94      0.96      0.95        71
         78       0.93      0.76      0.84        34
         79       0.70      0.85      0.77        33
         80       0.76      0.86      0.81        72
         81       0.93      0.98      0.95       132
         82       0.87      0.63      0.74       175
         83       0.78      0.73      0.75        48
         84       0.45      0.49      0.47        45
         85       1.00      0.96      0.98       110
         86       0.67      0.56      0.61        98
         87       0.80      0.73      0.77        83
         88       0.83      0.91      0.87        43
         89       0.84      0.91      0.88        35
         90       0.79      0.82      0.81        38
         91       0.65      0.55      0.60        51
         92       0.33      0.49      0.39        41
         93       0.86      0.92      0.89        48
         94       0.96      0.92      0.94        71
         95       0.84      0.82      0.83       117
         96       0.69      0.89      0.78        38
         97       0.68      0.81      0.74        31
         98       0.95      0.88      0.91        99
         99       0.77      0.77      0.77        69
        100       0.89      0.91      0.90        99
        101       0.65      0.61      0.63        85
        102       0.58      0.66      0.62        56
        103       0.94      0.94      0.94        36
        104       0.72      0.76      0.74        41
        105       0.72      0.65      0.68        65
        106       0.72      0.81      0.76        32
        107       0.93      0.77      0.84        35
        108       0.69      0.72      0.71        82
        109       0.94      0.82      0.88        78
        110       0.98      0.95      0.97       121
        111       0.68      0.78      0.72        54
        112       0.97      0.97      0.97        89
        113       0.73      0.97      0.83        31
        114       0.87      0.88      0.88       127
        115       0.96      0.94      0.95        78
        116       0.62      0.48      0.54       108
        117       0.85      0.88      0.87       110
        118       0.54      0.43      0.48        88
        119       0.90      0.77      0.83       107
        120       0.57      0.60      0.58        70
        121       0.77      0.95      0.85        39
        122       0.96      0.85      0.91       124
        123       0.88      0.76      0.82        59
        124       0.92      0.96      0.94        70
        125       0.76      0.87      0.81        39
        126       0.83      0.75      0.79       114
        127       1.00      1.00      1.00        56
        128       0.68      0.87      0.76        83
        129       0.97      0.91      0.94       125
        130       0.78      0.83      0.81        48
        131       0.85      0.88      0.86       112
        132       1.00      0.97      0.98        97
        133       0.99      0.75      0.85       164
        134       0.45      0.59      0.51        32
        135       0.83      0.88      0.85        81
        136       0.75      0.74      0.74       104
        137       0.72      0.72      0.72       134
        138       0.98      0.93      0.95        88
        139       0.33      0.13      0.19        38
        140       0.98      0.91      0.94        44
        141       0.87      0.84      0.85        62
        142       1.00      0.92      0.96       104
        143       0.63      0.73      0.67        51
        144       0.58      0.62      0.60        73
        145       0.70      0.86      0.77        50
        146       0.86      0.80      0.83        86
        147       0.88      0.87      0.88        93
        148       0.94      0.79      0.86        58
        149       0.92      0.90      0.91       110
        150       0.79      0.79      0.79        77
        151       0.66      0.63      0.64        75
        152       0.95      0.96      0.95        54
        153       0.49      0.94      0.65        32
        154       0.77      0.78      0.78        74
        155       0.89      0.74      0.81       189
        156       0.66      0.49      0.56        92
        157       0.88      0.94      0.91        31
        158       0.77      0.72      0.74       101
        159       0.55      0.56      0.55       109
        160       0.19      0.26      0.22        39
        161       0.68      0.75      0.72        57
        162       0.89      0.97      0.93        33
        163       0.41      0.41      0.41        70
        164       0.96      0.86      0.91       160
        165       0.38      0.40      0.39        40
        166       0.91      0.93      0.92        72
        167       0.76      0.84      0.80        64
        168       0.86      0.75      0.80        76
        169       0.75      0.82      0.79        40
        170       0.99      0.96      0.97       141
        171       0.18      0.06      0.10        62
        172       0.86      0.88      0.87        89
        173       0.65      0.72      0.68        39
        174       0.74      0.92      0.82        38
        175       0.80      0.96      0.87        45
        176       0.41      0.60      0.49        43
        177       0.70      0.88      0.78        32
        178       0.94      0.91      0.93        55
        179       0.48      0.62      0.54        32
        180       0.68      0.66      0.67        82
        181       0.58      0.59      0.58        32
        182       0.75      0.68      0.71        87
        183       0.67      0.86      0.75        42
        184       0.88      0.88      0.88        58
        185       0.72      0.87      0.79        30
        186       0.96      0.95      0.95        56
        187       0.68      0.97      0.80        31
        188       0.97      1.00      0.99        33
        189       0.77      0.91      0.83        47
        190       0.96      0.93      0.94       108
        191       0.98      1.00      0.99        40
        192       0.97      0.94      0.95       119
        193       0.97      1.00      0.99        33
        194       0.76      0.81      0.79       152
        195       0.95      0.83      0.89        71
        196       0.92      0.92      0.92        39
        197       0.95      0.94      0.95       124
        198       0.86      0.81      0.83        31
        199       0.48      0.55      0.51        38
        200       0.94      0.97      0.96        34
        201       0.32      0.39      0.35        31
        202       0.82      0.85      0.83        53
        203       0.89      0.91      0.90        34
        204       0.60      0.82      0.70        39
        205       0.64      0.72      0.67        75
        206       0.99      0.89      0.94       101
        207       0.33      0.36      0.34        45
        208       0.81      0.72      0.76       108
        209       0.90      0.94      0.92        80
        210       0.82      0.85      0.84        33
        211       0.88      0.95      0.91        60
        212       0.75      0.68      0.71        93
        213       0.83      0.77      0.80        87
        214       0.24      0.22      0.23        36
        215       0.48      0.49      0.48        92
        216       0.76      0.83      0.79        75
        217       0.55      0.59      0.57        71
        218       0.54      0.64      0.59        56
        219       0.90      0.78      0.84        60
        220       0.81      0.86      0.83        50
        221       0.57      0.65      0.61        43
        222       1.00      1.00      1.00        46
        223       0.73      0.71      0.72        97
        224       1.00      0.84      0.91        85
        225       0.46      0.73      0.56        33
        226       0.91      1.00      0.96        43
        227       0.96      0.94      0.95        54
        228       0.50      0.80      0.62        30
        229       0.35      0.69      0.47        32
        230       0.61      0.59      0.60        37
        231       0.94      0.83      0.88        59
        232       0.92      0.86      0.89        98
        233       0.96      0.93      0.94        70
        234       0.97      0.97      0.97        69
        235       0.92      0.86      0.89       104
        236       1.00      0.95      0.97        99
        237       0.92      0.87      0.90       123
        238       0.91      0.90      0.91        93
        239       1.00      0.96      0.98        48
        240       0.58      0.63      0.60        82
        241       0.76      0.85      0.80        33
        242       0.83      0.85      0.84        95
        243       0.90      0.90      0.90        63
        244       0.43      0.54      0.48        41
        245       0.79      0.70      0.74       104
        246       0.86      0.70      0.77        77
        247       0.95      1.00      0.98        62
        248       0.97      0.95      0.96        78
        249       0.85      0.75      0.80        61
        250       0.91      0.85      0.88       121
        251       0.61      0.55      0.58        31
        252       0.75      0.95      0.84        42
        253       0.61      0.67      0.64        45
        254       0.25      0.31      0.28        48
        255       1.00      1.00      1.00        42
        256       1.00      0.98      0.99        98
        257       0.88      0.94      0.91        32
        258       0.45      0.53      0.49        55
        259       0.93      0.96      0.95       128
        260       0.67      0.50      0.57       121
        261       0.59      0.85      0.69        39
        262       0.98      1.00      0.99        62
        263       0.50      0.69      0.58        64
        264       0.93      0.93      0.93        45
        265       0.86      0.96      0.91        56
        266       0.90      0.94      0.92        68
        267       0.87      0.87      0.87        67
        268       0.83      0.80      0.82       148
        269       0.64      0.81      0.71        31
        270       0.62      0.41      0.50       150
        271       0.93      0.88      0.90        43
        272       0.98      0.95      0.97       120
        273       0.94      1.00      0.97        44
        274       0.78      0.77      0.78       168
        275       0.33      0.66      0.44        38
        276       0.25      0.51      0.33        35
        277       0.86      0.97      0.91        58
        278       0.66      0.76      0.70        33
        279       0.54      0.57      0.55        63
        280       0.95      0.88      0.91        67
        281       0.64      0.84      0.73        45
        282       0.81      0.94      0.87        49
        283       0.80      1.00      0.89        32
        284       0.72      0.61      0.66        96
        285       0.77      0.81      0.79       118
        286       0.86      0.86      0.86        66
        287       0.77      0.67      0.72       107
        288       0.94      0.94      0.94        35
        289       0.94      0.88      0.91        88
        290       0.84      0.80      0.82        66
        291       0.99      0.88      0.93        96
        292       0.99      0.83      0.90       135
        293       0.71      0.84      0.77        44
        294       0.85      0.91      0.88        32
        295       0.88      0.88      0.88        57
        296       0.63      0.59      0.61        82
        297       0.90      0.84      0.87       107
        298       0.78      0.76      0.77        41
        299       0.83      0.81      0.82        31
        300       0.77      0.88      0.82        42
        301       0.73      0.83      0.78        53
        302       0.95      0.93      0.94        90
        303       0.56      0.77      0.65        35
        304       0.89      0.98      0.93        42
        305       0.95      0.89      0.92       102
        306       1.00      0.95      0.97        39
        307       0.88      0.77      0.82       120
        308       0.96      0.99      0.97       117
        309       0.90      0.86      0.88        98
        310       0.97      0.99      0.98       110
        311       0.45      0.50      0.47        36
        312       0.96      0.99      0.98        99
        313       0.79      0.95      0.86       116
        314       0.60      0.74      0.67        39
        315       0.58      0.70      0.64        30
        316       0.94      0.83      0.88       156
        317       0.77      0.84      0.80        75
        318       0.66      0.72      0.69        72
        319       0.98      0.90      0.94       154
        320       0.94      0.86      0.90        86
        321       0.26      0.17      0.21        59
        322       0.74      0.80      0.77       119
        323       1.00      1.00      1.00        44
        324       0.94      0.91      0.93        56
        325       0.56      0.56      0.56       112
        326       0.92      0.89      0.91       114
        327       0.86      0.85      0.85        93
        328       0.76      0.74      0.75        85
        329       0.39      0.21      0.27        76
        330       0.65      0.67      0.66        66
        331       0.86      0.90      0.88       128
        332       0.86      0.91      0.88        33
        333       0.49      0.55      0.52        56
        334       0.62      0.44      0.51        41
        335       0.87      0.65      0.74       103
        336       0.80      0.85      0.82        46
        337       0.96      0.93      0.94        54
        338       0.80      0.94      0.87        52
        339       0.73      0.67      0.70        57
        340       0.84      0.75      0.79       128
        341       0.91      0.93      0.92        42
        342       0.80      0.69      0.74       118
        343       0.53      0.58      0.55        45
        344       0.59      0.76      0.67        38
        345       0.99      0.97      0.98       153
        346       0.60      0.78      0.68        65
        347       0.91      0.83      0.87        64
        348       0.74      0.56      0.64        99
        349       1.00      0.99      1.00       105
        350       0.78      0.80      0.79        86
        351       0.95      0.96      0.96       125
        352       0.43      0.36      0.39        64
        353       0.42      0.56      0.48        50
        354       0.44      0.68      0.54        41
        355       0.93      0.87      0.90       108
        356       0.80      0.75      0.77        48
        357       0.73      0.65      0.69        98
        358       0.64      0.63      0.64        43
        359       0.73      0.75      0.74       120
        360       1.00      0.99      1.00       103
        361       0.44      0.45      0.44       120
        362       0.98      0.89      0.93       158
        363       0.89      0.98      0.93        51
        364       0.56      0.76      0.65        42
        365       0.88      0.84      0.86       122
        366       0.70      0.58      0.63       110
        367       1.00      0.94      0.97        67
        368       0.84      0.86      0.85       114
        369       0.68      0.84      0.75        62
        370       0.90      0.93      0.92       130
        371       0.67      0.73      0.70        62
        372       0.82      0.83      0.82       110
        373       0.89      0.90      0.89       145
        374       0.49      0.75      0.59        32
        375       0.49      0.36      0.41       100
        376       0.95      0.97      0.96        39
        377       0.70      0.74      0.72        38
        378       0.36      0.58      0.44        31
        379       0.80      0.81      0.81        54
        380       0.72      0.89      0.80        44
        381       0.35      0.40      0.37        48
        382       0.70      0.84      0.76        31
        383       0.74      0.79      0.76        89
        384       0.98      0.98      0.98       100
        385       0.93      0.91      0.92        46
        386       0.74      0.82      0.78        61
        387       0.60      0.75      0.67        32
        388       0.68      0.67      0.68       115
        389       0.74      0.86      0.80        43
        390       0.99      0.97      0.98       158
        391       0.67      0.72      0.69        89
        392       0.84      0.96      0.90        50
        393       0.93      0.93      0.93        69
        394       0.81      0.87      0.84        55
        395       0.79      0.86      0.82        81
        396       0.73      0.75      0.74        59
        397       0.94      0.97      0.96       121
        398       0.88      0.94      0.91        80
        399       0.83      0.87      0.85        45
        400       0.59      0.53      0.56        38
        401       0.78      0.84      0.81       108
        402       0.91      0.91      0.91        34
        403       0.90      0.84      0.87       123
        404       0.85      0.93      0.89        55
        405       0.88      0.84      0.86       103
        406       0.80      0.86      0.83       111
        407       0.72      0.76      0.74        76
        408       0.61      0.90      0.73        30
        409       0.54      0.60      0.57        42
        410       0.78      0.44      0.57        63
        411       0.84      0.76      0.80       136
        412       0.60      0.53      0.56        64
        413       0.97      0.97      0.97        32
        414       0.95      0.92      0.94        66
        415       0.38      0.38      0.38        88
        416       0.89      0.70      0.78       116
        417       0.94      0.94      0.94       124
        418       0.91      0.72      0.81       127
        419       0.84      0.89      0.87        76
        420       0.85      0.92      0.88        48
        421       0.88      1.00      0.94        30
        422       0.66      0.79      0.72        42
        423       0.75      0.70      0.73       110
        424       0.71      0.97      0.82        30
        425       0.77      0.89      0.82        37
        426       0.68      0.78      0.72        32
        427       0.75      0.94      0.83        32
        428       0.48      0.67      0.56        55
        429       0.96      0.98      0.97        88
        430       1.00      0.89      0.94        57
        431       0.98      1.00      0.99        43
        432       0.78      0.65      0.71        71
        433       0.47      0.51      0.49        55
        434       0.91      0.94      0.92        32
        435       0.93      0.82      0.87        66
        436       0.92      0.88      0.90        65
        437       0.75      0.94      0.83        32
        438       0.77      0.69      0.73        54
        439       0.79      0.73      0.76        30
        440       0.98      0.99      0.98        86
        441       0.39      0.64      0.48        33
        442       0.80      0.77      0.79       131
        443       0.54      0.65      0.59        86
        444       0.50      0.46      0.48        48
        445       0.76      0.97      0.85        38
        446       0.96      0.99      0.98       123
        447       0.59      0.73      0.65        63
        448       0.85      0.76      0.80        58
        449       0.83      0.95      0.89        57
        450       0.60      0.71      0.65        56
        451       0.68      0.78      0.73        51
        452       0.53      0.76      0.63        41
        453       0.97      0.93      0.95       137
        454       0.95      0.92      0.94        39
        455       0.79      0.89      0.84        63
        456       0.91      0.82      0.86        85
        457       0.91      0.78      0.84       137
        458       0.76      0.81      0.78        47
        459       0.62      0.41      0.50        87
        460       0.99      0.99      0.99        95
        461       0.49      0.74      0.59        46
        462       0.57      0.54      0.55        69
        463       0.89      0.72      0.80       120
        464       0.88      0.84      0.86        93
        465       0.75      0.90      0.82        63
        466       0.42      0.56      0.48        32
        467       0.77      0.70      0.73        73
        468       0.91      0.98      0.94        42
        469       0.92      0.87      0.89        78
        470       1.00      0.97      0.99        36
        471       0.81      0.66      0.73        44
        472       0.68      0.66      0.67       112
        473       0.83      0.94      0.88        36
        474       0.53      0.47      0.50        79
        475       0.47      0.74      0.58        31
        476       0.99      0.90      0.94        78
        477       0.93      0.83      0.88       107
        478       0.90      0.58      0.70        33
        479       0.87      0.94      0.90        63
        480       0.91      0.95      0.93        91
        481       0.73      0.76      0.74        46
        482       0.91      0.88      0.89        33
        483       0.99      0.86      0.92       167
        484       1.00      0.97      0.99        35
        485       0.84      0.75      0.79        36
        486       0.30      0.31      0.31        74
        487       0.76      0.78      0.77        60
        488       0.57      0.68      0.62        31
        489       0.92      0.96      0.94        49
        490       0.82      0.66      0.73       121
        491       0.82      0.85      0.84        39
        492       0.75      0.75      0.75       114
        493       0.92      0.85      0.89       149
        494       0.64      0.76      0.70        51
        495       0.91      0.93      0.92        80
        496       0.81      0.97      0.88        39
        497       0.62      0.48      0.54        88

avg / total       0.80      0.80      0.80     35746

"""