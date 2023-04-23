import os
import datetime

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#import IPython
#import IPython.display

#def load_data(data):


location = "file:///Users/Derpy/Documents/SecureWaterFuture/SWFdata/BearCreek_precipitation.csv"
PClocation = "file://" + "/Documents/Repositories/SecureWaterFuture/SWFdata/BearCreek_precipitation.csv"
data = pd.read_csv(PClocation)

#flood = []
#for i in range(len(data['Var1'])):
#    if(data['prdaily'][i] > 12.0):
#        flood.append(1)
#        print(data['Var1'][i])
#    else:
#        flood.append(0)
#data['flood'] = flood

data['flood'] = np.where(data['prdaily'] > 12, 1, 0) 

train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)

print(len(train))
print(len(test))

def create_dataset(data, shuffle=True, batch_size=32):
    df = data.copy()
    label = df.pop('flood')

    df = {key: value[:,tf.newaxis] for key, value in data.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds
    
batch_size = 5
#train_ds = create_dataset(train, batch_size=batch_size)
#train['prdaily'] = np.asarray(train['prdaily']).astype(np.float32)


train_ds = tf.convert_to_tensor(train['prdaily'], dtype=tf.float32)
train_ds = tf.convert_to_tensor(train)

    


