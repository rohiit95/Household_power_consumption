# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import scale

df=pd.read_csv('D:/ml/household_power_consumption.txt', delimiter=';')

df = df[(df.Global_active_power!="?")]

df=df[:20000]


df["Timestamp"] = df["Date"].astype(str)+" "+df["Time"]



df.Timestamp = pd.to_datetime(df.Timestamp, format = "%d/%m/%Y %H:%M:%S")



df.Date = pd.to_datetime(df.Date, format = "%d/%m/%Y")



df.Global_active_power = df.Global_active_power.astype(float)
df.Global_reactive_power = df.Global_reactive_power.astype(float)
df.Voltage = df.Voltage.astype(float)
df.Global_intensity=df.Global_intensity.astype(float)
df.Sub_metering_1 = df.Sub_metering_1.astype(float)
df.Sub_metering_2 = df.Sub_metering_2.astype(float)
df.Sub_metering_3 = df.Sub_metering_3.astype(float)



day1 = df[(df.Date == "2007-02-01")]


# plt.figure(figsize=(8,8))
#plt.plot(day1['Timestamp'], day1['Global_active_power'])
#plt.xticks(rotation='vertical')
    
gi=np.array(df['Global_intensity'])
gi=scale(gi)
gi=gi.reshape((gi.shape[0],1))

grp=np.array(df['Global_reactive_power'])
grp=scale(grp)
grp=grp.reshape((grp.shape[0],1))

volt=np.array(df['Voltage'])
volt=scale(volt)
volt=volt.reshape((volt.shape[0],1))

sub_meter1=np.array(df['Sub_metering_1'])
sub_meter1=scale(sub_meter1)
sub_meter1=sub_meter1.reshape((sub_meter1.shape[0],1))
sub_meter2=np.array(df['Sub_metering_2'])
sub_meter2=scale(sub_meter2)
sub_meter2=sub_meter2.reshape((sub_meter2.shape[0],1))
sub_meter3=np.array(df['Sub_metering_3'])
sub_meter3=scale(sub_meter3)
sub_meter3=sub_meter3.reshape((sub_meter3.shape[0],1))

#plt.plot(df['Timestamp'], df['Global_active_power'])
#plt.xticks(rotation='vertical')

gap=np.array(df.Global_active_power)



X=np.concatenate((gi,grp,volt,sub_meter1,sub_meter2,sub_meter3), axis=1)

X_train=X[:18000]

X_test=X[18000:20000]

clf= svm.SVR(kernel='rbf',C=10, gamma=10)

gap_train=gap[:18000]
gap_test=gap[18000:20000]

clf.fit(X_train,gap_train)

pred= clf.predict(X_test)

loss=np.sqrt(np.mean(np.square(pred-gap_test)))
accr=np.mean(abs(pred-gap_test)<1)