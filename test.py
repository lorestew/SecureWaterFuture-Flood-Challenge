import os
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

#import IPython
#import IPython.display

#import seaborn as sns

location = "file:///Users/Derpy/Documents/SecureWaterFuture/SWFdata/BearCreek_precipitation_future.csv"
PClocation = "file://" + "/Documents/Repositories/SecureWaterFuture/SWFdata/BearCreek_precipitation_future.csv"

train = pd.read_csv(    #reading csv file to pandas
    PClocation,
)


dates = pd.to_datetime( #parsing data to pandas
    train.pop("Var1"),
    dayfirst=True,
    format='mixed')
#print(train.head())

#plot_cols = ['Var2_1', 'Var2_2', 'Var2_3']  #plotting data against dates
#plot_features = train[plot_cols]
#plot_features.index = dates
#_ = plot_features.plot(subplots=True)
#
#plot_features = train[plot_cols][:365]
#plot_features.index = dates[:365]
#_ = plot_features.plot(subplots=True)
#plt.show()

print(train.describe().transpose()) #statistics

plt.hist2d(
    train['Var2_1'], 
    train['Var2_14'], 
    bins = (50,50), 
    vmax=20,
    range = [[0,1], [0,1]]
)
plt.colorbar()
plt.xlabel('Variable 2_1')
plt.ylabel('Variable 2_2')
plt.show()