from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import LocallyConnected2D
from keras.utils import np_utils
from keras.regularizers import l1, activity_l1, l1l2
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.optimizers import RMSprop
import numpy
import pandas
import csv
from keras.callbacks import EarlyStopping, ModelCheckpoint
K.set_image_dim_ordering('th')
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("train.csv", header=None)
dataset_train = dataframe.values

dataframe2 = pandas.read_csv("test.csv", header=None)
dataset_test = dataframe2.values

X_train = dataset_train[0:42000:,1:785].astype(float)
Y_train = dataset_train[0:42000:,0:1].astype(float)

X_test = dataset_train[32000:42000:,1:785]
Y_test = dataset_train[32000:42000:,0:1]

Z = dataset_test[0:28000:,0:784]

# Ckecking class imbalances : OK
"""
nombre = [0,0,0,0,0,0,0,0,0,0]
for i in range(42000):
	nombre[Y_train[i]]=nombre[Y_train[i]]+1

for i in range(10):
	print(nombre[i])
"""

# reshape to be [samples][pixels][width][height]
Z = Z.reshape(Z.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
Z = Z / 255

# one hot encode outputs
#Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = [Sequential(), Sequential(),Sequential(), Sequential(), Sequential()]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

j = 0

for i in range(1):
	for train, test in kfold.split(X_train, Y_train):
		print(j) 
		print("ieme fold")
		# create model
		# create model
		
		model[j].add(Convolution2D(32, 6, 6, border_mode='valid', input_shape=(1, 28, 28), activation='relu', W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
		model[j].add(MaxPooling2D(pool_size=(2, 2)))  
		model[j].add(Convolution2D(128, 4, 4, activation='relu'))
		model[j].add(MaxPooling2D(pool_size=(2, 2))) # Sous echantillonage 1/2
		model[j].add(Dropout(0.4))
		model[j].add(Flatten())
		model[j].add(Dense(300, activation ='relu'))#, W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
		model[j].add(Dropout(0.4))
		#model.add(Dense(60, activation ='relu'))
		#model.add(Dense(60, activation ='relu'))

		model[j].add(Dense(num_classes, activation='softmax'))

		model[j].compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
		#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
		#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent

		print (model[j].summary()) # Affiche les details du reseau !

		# reshape to be [samples][pixels][width][height]
		X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

		Y3 = np_utils.to_categorical(Y_train) # convert class vectors to binary class matrices
		
	
		
		# Early stopping.
		
		callbacks = [
    	EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    	ModelCheckpoint("/home/tom/Documents/Projets python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
		]
		
		
		# Fit the model		
		model[j].fit(X_train[train], Y3[train],validation_data=(X_train[test], Y3[test]), nb_epoch=40, batch_size=96, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)

		#model.fit(X2[train], Y3[train], nb_epoch=200, batch_size=96)

		# evaluate the model
		scores = model[0].evaluate(X_train[test], Y3[test])
		print("%s: %.2f%%" % (model[0].metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		j = j +1;

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


Z0_result = model[0].predict(Z, batch_size=32, verbose=0)
Z1_result = model[1].predict(Z, batch_size=32, verbose=0)
Z2_result = model[2].predict(Z, batch_size=32, verbose=0)
Z3_result = model[3].predict(Z, batch_size=32, verbose=0)
Z4_result = model[4].predict(Z, batch_size=32, verbose=0)

c = csv.writer(open("Submission.csv", "wb"))
c.writerow(["ImageId","Label"])
for i in range(28000):
	j_max = 0;
	max = Z0_result[i][0]+Z1_result[i][0]+Z2_result[i][0]+Z3_result[i][0]+Z4_result[i][0]
	for j in range(10):
		if (Z_restult[i][j]>max):
			max = Z0_result[i][j]+Z1_result[i][j]+Z2_result[i][j]+Z3_result[i][j]+Z4_result[i][j]
			j_max = j
	c.writerow([i+1, j_max])
