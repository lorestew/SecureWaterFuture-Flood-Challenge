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


day = 24*60*60
year = (365.2425)*day

timestamp_s = dates.map(pd.Timestamp.timestamp)
train['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
train['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
train['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
train['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(train['Day sin'])[:25])
plt.plot(np.array(train['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')



