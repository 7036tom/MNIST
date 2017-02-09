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
from keras import backend as K
from keras.optimizers import RMSprop
import numpy
import pandas
import csv
K.set_image_dim_ordering('th')
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("train.csv", header=None)
dataset_train = dataframe.values

dataframe2 = pandas.read_csv("test.csv", header=None)
dataset_test = dataframe2.values

X_train = dataset_train[0:42000:,1:785]
Y_train = dataset_train[0:42000:,0:1]

X_test = dataset_train[32000:42000:,1:785]
Y_test = dataset_train[32000:42000:,0:1]

Z = dataset_test[0:28000:,0:784]

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
Z = Z.reshape(Z.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
Z = Z / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# create model
model = Sequential()
model.add(Convolution2D(32, 6, 6, border_mode='valid', input_shape=(1, 28, 28), activation='relu', W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Convolution2D(128, 4, 4, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Sous echantillonage 1/2
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(300, activation ='relu'))#, W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
model.add(Dropout(0.4))
#model.add(Dense(60, activation ='relu'))
#model.add(Dense(60, activation ='relu'))

model.add(Dense(num_classes, activation='softmax'))

# Compile model
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])


# Affiche les details du reseau !
print (model.summary()) 

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=30, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


Z_restult = model.predict(Z, batch_size=32, verbose=0)

c = csv.writer(open("Submission.csv", "wb"))
c.writerow(["ImageId","Label"])
for i in range(28000):
	j_max = 0;
	max = Z_restult[i][0]
	for j in range(10):
		if (Z_restult[i][j]>max):
			max = Z_restult[i][j]
			j_max = j
	c.writerow([i+1, j_max])
