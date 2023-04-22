import os
import datetime

import tensorflow as tf
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
data = pd.read_csv(    #reading csv file to pandas
    location,
)

data.isna().sum()
data = data.dropna()

flood = []
for i in range(len(data['Var1'])):
    if(data['prdaily'][i] > 12.0):
        flood.append(1)
        print(data['Var1'][i])
    else:
        flood.append(0)
data['flood'] = flood

print(data.describe().transpose()) #statistics


train_data = data.sample(frac=.8, random_state=0)
test_data = data.drop(train_data.index)


sns.pairplot(
    train_data[['Var1', 'prdaily', 'flood']], 
    diag_kind='kde')
plt.show()

train_data.describe().transpose()

train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('flood')
test_labels = test_features.pop('flood')

train_data.describe().transpose()[['mean', 'std']]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())










#column_indices = {name: i for i, name in enumerate(data.columns)}
#
#
#n = len(data)
#train_data = data[0:int(n*.7)]
#val_data = data[int(n*.7):int(n*.9)]
#test_data = data[int(n*.9):]
#
#num_features = data.shape[1]
#
#
#train_mean = train_data.mean()
#train_std = train_data.std()
#
##normalizing data
#train_data = (train_data - train_mean) / train_std
#val_data = (val_data - train_mean) / train_std
#test_data = (test_data - train_mean) / train_std
#
#data_std = (data - train_mean) / train_std
#data_std = data_std.melt(var_name='Column', value_name='Normalized')
#plt.figure(figsize=(12, 6))
#ax = sns.violinplot(x='Column', y='Normalized', data=data_std)
#_ = ax.set_xticklabels(data.keys(), rotation=90)
#plt.show()






