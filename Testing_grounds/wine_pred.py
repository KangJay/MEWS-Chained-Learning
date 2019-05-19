from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#pip install -q -U tensorflow==1.7.0

import itertools
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

layers = keras.layers

#Download data over from a cloud storage
URL = "https://storage.googleapis.com/sara-cloud-ml/wine_data.csv"
path = tf.keras.utils.get_file(URL.split("/")[-1], URL)

#Holds our data set
data = pd.read_csv(path)

#Shuffle the data set
data = data.sample(frac=1)

#Preprocess the data by adding a threshhold to filter out results
data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1)

#anything less than 500 will be dropped.
variety_threshold = 500
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove, np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]

#Split the data into training and test sets
train_size = int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))

#Training features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

#Training labels
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

#Testing labels
labels_test = data['price'][train_size:]

#create a tokenizer to preprocess our text descriptions
#So after this step, we extract the top 12000 most utilized words
vocab_size = 12000 #Hyperparameter for our dataset. Top 12000 words for our dataset's descriptions
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train) #Only fit on training

#Wide feature 1: creating a sparse bag of words (bow) vocab_size vector
#This step takes up A LARGE AMOUNT OF RAM ~2.5 GB
description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)

#Use sklearn utility to convert label strings to numbered index
#There's around 632 varieties of wine, so we do another filtering to get only the top 40 varieties.
#then convert each variety to a integer representation of it
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1

#convert labels to one hot
variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variet_test, num_classes)
