import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

import sys

train_df=pd.read_csv('train.csv')
valid_df=pd.read_csv('test.csv')
IMG_SIZE = (224, 224)

batch_input=int(sys.argv[1])


def data_gen(batch,image_size,train_df_input,valid_df_input):

	train_idg=ImageDataGenerator(rescale=1./255.0,horizontal_flip=True, vertical_flip=False,
		height_shift_range=0.1, width_shift_range=0.1, rotation_range=20, shear_range=0.1,zoom_range=0.1)

	train_gen=train_idg.flow_from_dataframe(dataframe=train_df_input,directory=None, x_col='img_path',y_col='class',
		class_mode='binary',target_size=image_size, batch_size=batch)

	val_idg=ImageDataGenerator(rescale=1./255.0)
	val_gen=val_idg.flow_from_dataframe(dataframe=valid_df_input,direcotory=None,x_col='img_path',y_col='class',
		class_mode='binary',target_size=image_size, batch_size=batch)

	X,Y= val_gen.next()
	return X,Y,train_gen


model=VGG16(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input,outputs=transfer_layer.output)

for layer in vgg_model.layers[0:17]:
	layer.trainable=False

for layer in vgg_model.layers:
	print(layer.name,layer.trainable)


def train_model(dropout_rate,input_model,activation_mode):
	new_model=Sequential()
	new_model.add(input_model)
	new_model.add(Flatten())
	new_model.add(Dropout(dropout_rate))
	new_model.add(Dense(1024,activation=str(activation_mode)))
	new_model.add(Dropout(dropout_rate))
	new_model.add(Dense(512,activation=str(activation_mode)))
	new_model.add(Dropout(dropout_rate))
	new_model.add(Dense(256,activation=str(activation_mode)))
	new_model.add(Dense(1,activation='sigmoid'))
	return new_model

activation_mode='relu'
droprate=float(sys.argv[2])
new=train_model(droprate,vgg_model,activation_mode)
X,Y,train_gen=data_gen(batch_input,[224,224],train_df,valid_df)

def plot_model(new_model,testX,testY,optimizer,loss,metrics,train_gen_input,epochs_input):
	new_model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
	history=new_model.fit_generator(train_gen_input,validation_data=(testX,testY),epochs=epochs_input)
	plot_history(history)
	plt.close('all')

def plot_history(input_his):
	N=len(input_his.history['loss'])
	plt.style.use('ggplot')
	plt.figure()
	x=np.arange(0,N)
	plt.plot(x,input_his.history['loss'],label='train_loss')
	plt.plot(x,input_his.history['val_loss'],label='val_loss')
	plt.plot(x,input_his.history['binary_accuracy'],label='train_accuracy')
	plt.plot(x,input_his.history['val_binary_accuracy'],label='val_accuracy')
	plt.title('Training Loss and Accuracy on Dataset')
	plt.xlabel("Epoch number as total")
	plt.ylabel('Loss v.s. Accuracy')
	plt.legend(loc='upper right')
	plt.show()

epochs_num=int(sys.argv[3])
learning_rate=float(sys.argv[4])
optimizer=Adam(lr=learning_rate)
loss='binary_crossentropy'
metrics=['binary_accuracy']

plot_model(new,X,Y,optimizer,loss,metrics,train_gen,epochs_num)




