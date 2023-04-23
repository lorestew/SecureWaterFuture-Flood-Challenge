import os
import datetime

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

location = "file:///Users/Derpy/Documents/SecureWaterFuture/SWFdata/BearCreek_precipitation.csv"
PClocation = "file://" + "/Documents/Repositories/SecureWaterFuture/SWFdata/BearCreek_precipitation.csv"

data = tf.keras.utils.get_file("BearCreek_precipitation.csv", location)
df = pd.read_csv(data)
print(df.head())
dataset = tf.data.Dataset.from_tensor_slices(dict(df))
#for feature_batch in data_slices:
#    for key, value in feature_batch.items():
#        print("    {!r:20s}:   {}".format(key, value))

features = df.copy()
labels = features.pop('flood')


features = np.array(features)
print(features)


rain_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

rain_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                   optimizer = tf.keras.optimizers.Adam())

rain_model.fit(features, labels, epochs=3)