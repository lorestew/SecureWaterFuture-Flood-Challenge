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

location = "file:///Users/Derpy/Documents/SecureWaterFuture/SWFdata/BearCreek_precipitation_future.csv"


train = pd.read_csv(
    location,
    names=["Date", "Stage Height", "Streamflow"]
)

print(train.head())

#dates = pd.to_datetime(train.pop(
#    "Date",
#    format ='%d.%m.%Y'))
#
#print(train.head())




