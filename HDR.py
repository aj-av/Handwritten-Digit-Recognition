import sys
import os
import time
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as NNlayer
from lasagne.nonlinearities import softmax
from array import array
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import easygui


# We are training with mini_batch gradient descent algorithm, so this function creates the batches of given size of data set
def new_batch(X, Y, bsize): 
    for i in range(0, len(X) - bsize + 1, bsize):
        indices = slice(i, i + bsize)
        yield X[indices], Y[indices]

# Function to create Input layer
def New_Input_Layer(inputs):
    shape = (None, 1, 28, 28)
    return NNlayer.InputLayer(shape=shape,input_var=inputs)

# Function to create hidden layer
def New_Hidden_Layer(prev_layer, hidden_units): # if first than initialize weights
    return NNlayer.DenseLayer(prev_layer, num_units=hidden_units)
      
# Function to create output layer
def New_Output_Layer(prev_layer, output_units):
    return NNlayer.DenseLayer(prev_layer, num_units=output_units, nonlinearity=softmax)

# Function to create convolution layer (used by CNN only)
def New_CONV_Layer(prev_layer, features, fsize): # features : number of filters
    return NNlayer.Conv2DLayer(prev_layer, num_filters=features, filter_size=(fsize,fsize))

# Function to create pool layer (used by CNN only)
def New_POOL_Layer(prev_layer, psize):
    return NNlayer.MaxPool2DLayer(prev_layer, pool_size=(psize, psize))


# Creates Neural Network [MLP or CNN]
def Create_NNet(inputs, model):
    if model=='mlp':  # CREATE MLP NEURAL NETWORK                        
        ip_layer     = New_Input_Layer(inputs)    # INPUT LAYER
        hid1_layer   = New_Hidden_Layer(ip_layer, 800) # HIDDEN LAYER 1
        hid2_layer   = New_Hidden_Layer(hid1_layer, 800) # HIDDEN LAYER 2
        NNet         = New_Output_Layer(hid2_layer, 10) # OUTPUT LAYER [Neural Network is reffered by this layer since it contains stack of all layers]
    else:             # CREATE CNN NEURAL NETWORK
        ip_layer   = New_Input_Layer(inputs) # INPUT LAYER
        conv1      = New_CONV_Layer(ip_layer, 32, 5) # CONVOLUTION LAYER 1, with 32 filters each of size 5x5
        max1       = New_POOL_Layer(conv1, 2) # MAX POOL LAYER OF CONVOLUTION LAYER 1
        conv2      = New_CONV_Layer(max1, 32, 5) # CONVOLUTION LAYER 2, with 32 filters each of size 5x5
        max2       = New_POOL_Layer(conv2, 2) # MAX POOL LAYER OF CONVOLUTION LAYER 2
        FC_layer   = New_Hidden_Layer(max2, 256) # LAST FULLY CONNECTED LAYER
        NNet       = New_Output_Layer(FC_layer, 10) # OUTPUT LAYER OF CNN
    return NNet
        
def imshow(digit, yh, y=None):
	plt.imshow(digit, cmap=plt.cm.gray_r, interpolation="nearest")
	if y:
		s = 'Actual: '+str(y)+'\nPredicted: '+str(yh)
	else:	
		s = 'Predicted: '+str(yh)
	plt.text(0.5, 2, s)
	plt.show()
def show_test(x, y=None):
	yh = pred_NNet(x)
	yh = np.argmax(yh)
	if y:
		imshow(x[0][0], y, yh)
	else:
		imshow(x[0][0], yh)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])




model=str(sys.argv[1])
if(model=='dtree'):
	
	# Read Training Images
	train_im_f = 'train-images.idx3-ubyte'
	train_lbl_f = 'train-labels.idx1-ubyte'

	file = open(train_im_f, 'rb')
	trainIm = file.read()
	file.close()

	trainIm = trainIm[16:]
	temp = array("B",trainIm)
	trainX = np.array(temp).reshape(60000, 28*28) # change if type is required to change

	# Read Training Image labels/target
	file = open(train_lbl_f, 'rb')
	trainLbl = file.read()
	file.close()

	trainLbl = trainLbl[8:]
	temp = array("B",trainLbl)
	trainY = np.array(temp)

	########################################
	# Read Testing Images

	test_im_f = 't10k-images.idx3-ubyte'
	test_lbl_f = 't10k-labels.idx1-ubyte'

	file = open(test_im_f, 'rb')
	testIm = file.read()
	file.close()

	testIm = testIm[16:]
	temp = array("B",testIm)
	testX = np.array(temp).reshape(10000, 28*28) # change if type is required to change

	# Read Training Image labels/target
	file = open(test_lbl_f, 'rb')
	testLbl = file.read()
	file.close()

	testLbl = testLbl[8:]
	temp = array("B",testLbl)
	testY = np.array(temp)

	print('Training Model...')
	clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
	clf.fit(trainX,trainY)

	print('Testing...')
	ans = clf.predict(testX[0:10000])

	msclfd = 0
	for i in range(10000):
		if (ans[i] != testY[i]):
			msclfd = msclfd +1
	
	print ('Accuracy : ', (10000-msclfd)/100,'%')

else:
	########################################
	# Read Testing Images

	test_im_f = 't10k-images.idx3-ubyte'
	test_lbl_f = 't10k-labels.idx1-ubyte'

	file = open(test_im_f, 'rb')
	testIm = file.read()
	file.close()

	testIm = testIm[16:]
	temp = array("B",testIm)
	X_test = np.array(temp).reshape(-1, 1,28, 28) # change if type is required to change
	X_test = X_test/np.float32(256)

	# Read Training Image labels/target
	file = open(test_lbl_f, 'rb')
	testLbl = file.read()
	file.close()

	testLbl = testLbl[8:]
	temp = array("B",testLbl)
	y_test = np.array(temp)



	if(len(sys.argv)>2):
		saved_model = str(sys.argv[2])
		model = str(sys.argv[1])
		
		inputs = T.tensor4('inputs') # input to neural network as batch of images [symbolic vairable]
		outputs = T.ivector('output') # prediimport matplotlib.image as mpimgction of neural network for whole batch [sybolic variable]

		NNet = Create_NNet(inputs, model)  #Neural Network
		weights = NNlayer.get_all_params(NNet, trainable=True) # Weight matrices of Neural network
		predict = NNlayer.get_output(NNet) # returns the output of the last layer of neural network
		pred_NNet = theano.function([inputs], predict)
		Err = lasagne.objectives.categorical_crossentropy(predict, outputs) # Error function to be minizied [here is cross entropy loss function]
		Err = Err.mean() 
		# To Update weight matrices of neural network using mini-batch gradient descent algorithm with momentum technique to optimize
		updates = lasagne.updates.nesterov_momentum(Err, weights, learning_rate=0.01, momentum=0.9) 

		# returns the accuracy obtained in prediction
		check_acc = T.mean(T.eq(T.argmax(predict, axis=1), outputs),dtype=theano.config.floatX) #

		# Training function which updates weight matrices of neural network by calling update function and Err function.
		train = theano.function([inputs, outputs], Err, updates=updates) #

		# Function to validate predicted and actual output, it returns prediction error and accuracy 
		validate = theano.function([inputs, outputs], [Err, check_acc]) #
		
		with np.load(saved_model) as f:
			param = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(NNet, param)
		
		
	else:
		iterations=1#300
		# Read Training Images
		train_im_f = 'train-images.idx3-ubyte'
		train_lbl_f = 'train-labels.idx1-ubyte'

		file = open(train_im_f, 'rb')
		trainIm = file.read()
		file.close()

		trainIm = trainIm[16:]
		temp = array("B",trainIm)
		X_train = np.array(temp).reshape(-1, 1, 28, 28) # change if type is required to change
		X_train = X_train/np.float32(256)
		X_val = X_train[-10000:]
		X_train = X_train[:-10000]

		# Read Training Image labels/target
		file = open(train_lbl_f, 'rb')
		trainLbl = file.read()
		file.close()

		trainLbl = trainLbl[8:]
		temp = array("B",trainLbl)
		y_train = np.array(temp)
		y_val = y_train[-10000:]
		y_train = y_train[:-10000]

	


		# Creating Neural Network
		inputs = T.tensor4('inputs') # input to neural network as batch of images [symbolic vairable]
		outputs = T.ivector('output') # prediimport matplotlib.image as mpimgction of neural network for whole batch [sybolic variable]

		NNet = Create_NNet(inputs, model)  #Neural Network
		weights = NNlayer.get_all_params(NNet, trainable=True) # Weight matrices of Neural network
		predict = NNlayer.get_output(NNet) # returns the output of the last layer of neural network
		pred_NNet = theano.function([inputs], predict)
		Err = lasagne.objectives.categorical_crossentropy(predict, outputs) # Error function to be minizied [here is cross entropy loss function]
		Err = Err.mean() 
		# To Update weight matrices of neural network using mini-batch gradient descent algorithm with momentum technique to optimize
		updates = lasagne.updates.nesterov_momentum(Err, weights, learning_rate=0.01, momentum=0.9) 

		# returns the accuracy obtained in prediction
		check_acc = T.mean(T.eq(T.argmax(predict, axis=1), outputs),dtype=theano.config.floatX) #

		# Training function which updates weight matrices of neural network by calling update function and Err function.
		train = theano.function([inputs, outputs], Err, updates=updates) #

		# Function to validate predicted and actual output, it returns prediction error and accuracy 
		validate = theano.function([inputs, outputs], [Err, check_acc]) #




		print("Training Model...")
		
		for itr in range(iterations):
			# In each itr, we do a full pass over the training data:
			Terr = 0 # training error 
			Verr = 0 # validation error
			Vacc = 0 # validation accuracy
			tb_num = 0  # batch number while training
			vb_num = 0  # batch number while validation
			t1 = time.time()
			for batch in new_batch(X_train, y_train, 500):
				X, Y = batch    
				Terr += train(X, Y) # trains NN with batch
				tb_num += 1
				# And a full pass over the validation data:
		
			for batch in new_batch(X_val, y_val, 500):
				X, Y = batch
				err, acc = validate(X, Y) # Performs validation
				Verr += err
				Vacc += acc
				vb_num += 1
			 # Then we print the results for this itr:
			print("itr ",itr+1,"of ",iterations," took ",time.time() - t1,"s")
			print("Training error:  ",(Terr / tb_num))
			print("Validation error:  ",(Verr / vb_num))
			print("Validation accuracy:  ",(Vacc / vb_num * 100),"%")
		 # After training, we compute and print the test error:
	Terr = 0 # Testing error
	Tacc = 0 # Testing accuracy
	tb_num = 0 # batch number
	print('Testing...')
	for batch in new_batch(X_test, y_test, 500):
		X, Y = batch
		err, acc = validate(X, Y)
		Terr += err
		Tacc += acc
		tb_num += 1
	print("\n\n\nFinal results:")
	print ('Accuracy : ',Tacc / tb_num * 100,'%')
	################

#	xx = X_train[353].reshape(-1, 1, 28, 28)	
#	yy = y_train[353]
	
#	show_test(xx, yy)
	fpath = easygui.fileopenbox()
	img=mpimg.imread(fpath)
	gray = rgb2gray(img)
	x = np.float32(1- np.array(gray)/np.float32(256))
	x = x.reshape(-1, 1, 28, 28)
	show_test(x)
