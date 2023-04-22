import os
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

#import IPython
#import IPython.display
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns

#zip_path = tf.keras.utils.get_file(
#    fname=r'data/BearCreek_McKee_flow.csv',
#    origin=None,
#    extract=False)
#csv_path, _ = os.path.splitext(zip_path)

#df = pd.read_csv(csv_path)

#csv_path = "data/BearCreek_McKee_flow.csv"
#stream_train = pd.read_csv(
#    csv_path,
#    names = ["date", "stage_height", "streamflow"])
#stream_train.head()

location = "file://C:/Users/Derpy/Documents/SecureWaterFuture/SWFdata/BearCreek_McKee_flow.csv"
path = tf.keras.utils.get_file(
    fname="BearCreek_McKee_flow",
    origin=location)

stream_train_tf = tf.data.experimental.make_csv_dataset(
    path,
    batchsize = 10,
    num_epochs=1,
    ignore_errors=True
)

for batch, label in stream_train_tf.take(1):
    for key, value in batch.items():
        print(f"{key:10s}: {value}")






