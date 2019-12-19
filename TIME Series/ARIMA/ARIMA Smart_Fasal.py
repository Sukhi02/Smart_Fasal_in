
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
  

import os
import numpy as np
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Error(y_true, y_pred): 
     return ((np.array(y_true)- np.array(y_pred)))
     
from sklearn.preprocessing import MinMaxScaler




from pandas.plotting import autocorrelation_plot
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf




os.chdir("D:/CSIR_Smart_Fasal/cODE/Smart_Fasal_in/TIME Series/ARIMA/SMART_FASAL")





################ Soil Moisture at 10 m #########################################






#raw_dataset['Timestamp'] = data_time


#raw_dataset.to_csv("Dataset.csv")


#index = data_time



#raw_dataset = read_csv('dataset.csv', parse_dates=["date_time"], index_col= "date_time")

filename = 'dataset.csv'

#raw_dataset.index = index
#raw_dataset = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','Day', 'Date', 'Time'])


raw_dataset = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings'])

data_time = pd.to_datetime(raw_dataset['Timestamp'], unit='s')


#raw_dataset['index'] = data_time

raw_dataset.index = data_time

raw_dataset = raw_dataset.drop(columns = ['Timestamp'], axis=1)

raw_dataset.to_csv("Soil moisture ARIMA.csv")


raw_dataset = pd.read_csv('Soil moisture ARIMA.csv', parse_dates=["Timestamp"], index_col= "Timestamp")
raw_dataset = raw_dataset.drop(['Battery','Readings'], axis =1)

dataset =               raw_dataset.iloc[:, 0:8].values
dataset_10 =            raw_dataset.iloc[:, 0:1].values
dataset_45 =            raw_dataset.iloc[:, 1:2].values
dataset_80 =            raw_dataset.iloc[:, 2:3].values
dataset_Temperature =   raw_dataset.iloc[:, 3:4].values
dataset_Humidity =      raw_dataset.iloc[:, 4:5].values
dataset_Pressure =      raw_dataset.iloc[:, 5:6].values
dataset_Lum =           raw_dataset.iloc[:, 6:7].values
   
    # split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size= len(dataset) - train_size
total_size = train_size+1

train_data_10 =             dataset_10[:train_size]
train_data_45 =             dataset_45[:train_size]
train_data_80 =             dataset_80[:train_size]
train_data_Temperature =    dataset_Temperature[:train_size]
train_data_Humidity =       dataset_Humidity[:train_size]
train_data_Pressure =       dataset_Pressure[:train_size]
train_data_Lum =            dataset_Lum[:train_size]
test_data_10  =             dataset_10[train_size+1: ]
test_data_45  =             dataset_45[train_size+1: ]
test_data_80  =             dataset_80[train_size+1: ]
test_data_Temperature  =    dataset_Temperature[train_size+1: ]
test_data_Humidity  =       dataset_Humidity[train_size+1: ]
test_data_Pressure  =       dataset_Pressure[train_size+1: ]
test_data_Lum  =            dataset_Lum[train_size+1: ]
    

raw_dataset.index = data_time

SM_10 = raw_dataset['S_M_10cm']
SM_10.plot()
plt.show()


plt.figure(1)
plt.subplot(211)
SM_10.hist()
plt.subplot(212)
SM_10.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(SM_10, ax=plt.gca())
plt.subplot(212)
plot_pacf(SM_10, ax=pyplot.gca())
plt.show()





history = [x for x in train_data_10]
predictions = list()
for t in range(len(test_data_10)):
	model = ARIMA(history, order=(1,0,4))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_10[t]
	history.append(obs)
	print('10m ',t ,' predicted=%f, expected=%f' % (yhat, obs))  
testPredict_10  = predictions  


#trainPredict_10 = sc.inverse_transform(Y_10)
#testPredict_10 = sc.inverse_transform(testPredict_10)
error_10 = mean_squared_error(test_data_10 , testPredict_10)
print('Test MSE: %.3f' % error_10)

MAE_10= mean_absolute_error(test_data_10 , testPredict_10)
print('Test MAE: %.3f' % MAE_10)

rmse_10 = sqrt(mean_squared_error(test_data_10 , testPredict_10))
print('Test RMSE: %.3f' % rmse_10)

MAPE_10= mean_absolute_percentage_error(test_data_10 , testPredict_10)
print("\n\n \tMean absolute percentage error  :",MAPE_10)
print(".......Accuracy for 80 m  ...", 100-MAPE_10)


# plot
plt.plot(test_data_10, c = 'green', label = 'Actual')
plt.plot(testPredict_10, c = 'red', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Moisture Level 10 cm')
plt.title('Forecasted moisture 10 cm')
plt.legend()
plt.savefig('Moisture Level at 10 ARIMA ')
plt.show()






################ Soil Moisture at 25 m #########################################


SM_45 = raw_dataset['S_M_45cm']
SM_45.plot()
plt.show()


plt.figure(1)
plt.subplot(211)
SM_45.hist()
plt.subplot(212)
SM_45.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(SM_45, ax=plt.gca())
plt.subplot(212)
plot_pacf(SM_45, ax=pyplot.gca())
plt.show()





history = [x for x in train_data_45]
predictions = list()
for t in range(len(test_data_45)):
	model = ARIMA(history, order=(1,0,3))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_45[t]
	history.append(obs)
	print(' 40',1200 -t ,'/ predicted=%f, expected=%f' % (yhat, obs))
testPredict_45  = predictions  


#trainPredict_25 = sc.inverse_transform(Y_25)
#testPredict_25 = sc.inverse_transform(testPredict_25)
error_45 = mean_squared_error(test_data_45 , testPredict_45)
print('Test MSE: %.3f' % error_45)

rmse_45 = sqrt(mean_squared_error(test_data_45 , testPredict_45))
print('Test RMSE: %.3f' % rmse_45)

MAE_45= mean_absolute_error(test_data_45 , testPredict_45)
print('Test MAE: %.3f' % MAE_45)


MAPE_45= mean_absolute_percentage_error(test_data_45 , testPredict_45)
print("\n\n \tMean absolute percentage error  :",MAPE_45)
print(".......Accuracy for 80 m  ...", 100-MAPE_45)

plt.plot(test_data_45, c = 'green', label = 'Actual')
plt.plot(testPredict_45, c = 'red', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Moisture Level 40 cm')
plt.title('Forecasted moisture 40 cm')
plt.legend()
plt.savefig('Moisture Level at 40 ARIMA ')
plt.show()



################ Soil Moisture at 50 m #########################################



SM_80 = raw_dataset['S_M_80cm']
SM_80.plot()
plt.show()


plt.figure(1)
plt.subplot(211)
SM_80.hist()
plt.subplot(212)
SM_80.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(SM_80, ax=plt.gca())
plt.subplot(212)
plot_pacf(SM_80, ax=pyplot.gca())
plt.show()




history = [x for x in train_data_80]
predictions = list()
for t in range(len(test_data_80)):
	model = ARIMA(history, order=(1,0,4))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_80[t]
	history.append(obs)
	print('50',t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_80  = predictions  


#trainPredict_50 = sc.inverse_transform(Y_50)
#testPredict_50 = sc.inverse_transform(testPredict_50)
error_80 = mean_squared_error(test_data_80 , testPredict_80)
print('Test MSE: %.3f' % error_80)

MAE_80= mean_absolute_error(test_data_80, testPredict_80)
print('Test MAE: %.3f' % MAE_80)

rmse_80 = sqrt(mean_squared_error(test_data_80, testPredict_80))
print('Test RMSE: %.3f' % rmse_80)

MAPE_80= mean_absolute_percentage_error(test_data_80 , testPredict_80)
print("\n\n \tMean absolute percentage error  :",MAPE_80)
print(".......Accuracy for 80 m  ...", 100-MAPE_80)

plt.plot(test_data_80, c = 'green', label = 'Actual')
plt.plot(testPredict_80, c = 'red', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Moisture Level 80 cm')
plt.title('Forecasted moisture 80 cm')
plt.legend()
plt.savefig('Moisture Level at 80 ARIMA ')
plt.show()


################ Soil Moisture at 80 m #########################################


Temperature_P = raw_dataset['Temperature']
Temperature_P.plot()
plt.show()


plt.figure(1)
plt.subplot(211)
Temperature_P.hist()
plt.subplot(212)
Temperature_P.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(Temperature_P, ax=plt.gca())
plt.subplot(212)
plot_pacf(Temperature_P, ax=pyplot.gca())
plt.show()



history = [x for x in train_data_Temperature]
predictions = list()
for t in range(len(test_data_Temperature)):
	model = ARIMA(history, order=(1,0,6))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_Temperature[t]
	history.append(obs)
	print('Temperature' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_Temperature  = predictions  


#trainPredict_80 = sc.inverse_transform(Y_80)
#testPredict_80 = sc.inverse_transform(testPredict_80)


error_Temperature = mean_squared_error(test_data_Temperature , testPredict_Temperature)
print('Test MSE: %.3f' % error_Temperature)


MAE_Temperature = mean_absolute_error(test_data_Temperature, testPredict_Temperature)
print('Test MAE: %.3f' % MAE_Temperature)


rmse_Temperature = sqrt(mean_squared_error(test_data_Temperature, testPredict_Temperature))
print('Test RMSE: %.3f' % rmse_Temperature)

MAPE_Temperature = mean_absolute_percentage_error(test_data_Temperature , testPredict_Temperature)
print("\n\n \tMean absolute percentage error  :",MAPE_Temperature)
print(".......Accuracy for 80 m  ...", 100-MAPE_Temperature)

plt.plot(testPredict_Temperature, c = 'red', label = 'Predicted')
plt.plot(test_data_Temperature , c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Forecasted Temperature')
plt.legend()
plt.savefig('Temperature ARIMA ')
plt.show()


################ Humidity  #########################################


Humidity_P = raw_dataset['Humidity']
Humidity_P.plot()
plt.show()


plt.figure(1)
plt.subplot(211)
Humidity_P.hist()
plt.subplot(212)
Humidity_P.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(Humidity_P, ax=plt.gca())
plt.subplot(212)
plot_pacf(Humidity_P, ax=pyplot.gca())
plt.show()






history = [x for x in train_data_Humidity]
predictions = list()
for t in range(len(test_data_Humidity)):
	model = ARIMA(history, order=(1,0,6))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_Humidity[t]
	history.append(obs)
	print('Humidity' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_Humidity  = predictions  


#trainPredict_80 = sc.inverse_transform(Y_80)
#testPredict_80 = sc.inverse_transform(testPredict_80)

error_Humidity = mean_squared_error(test_data_Humidity , testPredict_Humidity)
print('Test MSE: %.3f' % error_Humidity)

MAE_Humidity= mean_absolute_error(test_data_Humidity, testPredict_Humidity)
print('Test MAE: %.3f' % MAE_Humidity)

from math import sqrt
# calculate RMSE
rmse_Humidity = sqrt(mean_squared_error(test_data_Humidity, testPredict_Humidity))
print('Test RMSE: %.3f' % rmse_Humidity)

MAPE_Humidity = mean_absolute_percentage_error(test_data_Humidity, testPredict_Humidity)
print("\n\n \tMean absolute percentage error  :",MAPE_Humidity)
print(".......Accuracy for 80 m  ...", 100-MAPE_Humidity)


plt.plot(testPredict_Humidity, c = 'red', label = 'Predicted')
plt.plot(test_data_Humidity , c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.title('Forecasted Humidity')
plt.legend()
plt.savefig('Humidity ARIMA ')
plt.show()




################ Pressure  #########################################



Pressure_P = raw_dataset['Pressure']
Pressure_P=Pressure_P.astype(float)
Pressure_P.plot()
plt.show()




plt.figure(1)
plt.subplot(211)
Pressure_P.hist()
plt.subplot(212)
Pressure_P.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(Pressure_P, ax=plt.gca())
plt.subplot(212)
plot_pacf(Pressure_P, ax=pyplot.gca())
plt.show()



history = [x for x in train_data_Pressure]
predictions = list()
for t in range(len(test_data_Pressure)):
	model = ARIMA(history, order=(1,0,6))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_Pressure[t]
	history.append(obs)
	print('Pressure' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_Pressure  = predictions  


#trainPredict_80 = sc.inverse_transform(Y_80)
#testPredict_80 = sc.inverse_transform(testPredict_80)

error_Pressure = mean_squared_error(test_data_Pressure , testPredict_Pressure)
print('Test MSE: %.3f' % error_Pressure)

MAE_Pressure= mean_absolute_error(test_data_Pressure, testPredict_Pressure)
print('Test MAE: %.3f' % MAE_Pressure)
# calculate RMSE
rmse_Pressure = sqrt(mean_squared_error(test_data_Pressure, testPredict_Pressure))
print('Test RMSE: %.3f' % rmse_Pressure)

MAPE_Pressure = mean_absolute_percentage_error(test_data_Pressure, testPredict_Pressure)
print("\n\n \tMean absolute percentage error  :",MAPE_Pressure)
print(".......Accuracy for 80 m  ...", 100-MAPE_Pressure)

plt.plot(testPredict_Pressure, c = 'red', label = 'Predicted')
plt.plot(test_data_Pressure, c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Forecasted Pressure')
plt.legend()
plt.savefig('Pressure ARIMA ')
plt.show()

################ Luminisity  #########################################



Luxes_P = raw_dataset['Luxes']
Luxes_P.plot()
plt.show()



plt.figure(1)
plt.subplot(211)
Luxes_P.hist()
plt.subplot(212)
Luxes_P.plot(kind='kde')
plt.show()

plt.figure()
plt.subplot(211)
plot_acf(Luxes_P, ax=plt.gca())
plt.subplot(212)
plot_pacf(Luxes_P, ax=pyplot.gca())
plt.show()



history = [x for x in train_data_Lum]
predictions = list()
for t in range(len(test_data_Pressure)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_data_Lum[t]
	history.append(obs)
	print('Lum' ,t ,' predicted=%f, expected=%f' % (yhat, obs))
testPredict_Lum  = predictions  



#trainPredict_80 = sc.inverse_transform(Y_80)
#testPredict_80 = sc.inverse_transform(testPredict_80)

error_Lum = mean_squared_error(test_data_Lum , testPredict_Lum)
print('Test MSE: %.3f' % error_Lum)

MAE_Lum = mean_absolute_error(test_data_Lum, testPredict_Lum)
print('Test MAE: %.3f' % MAE_Lum)

# calculate RMSE
rmse_Lum = sqrt(mean_squared_error(test_data_Lum, testPredict_Lum))
print('Test RMSE: %.3f' % rmse_Lum)


MAPE_Lum = mean_absolute_percentage_error(test_data_Lum, testPredict_Lum)
print("\n\n \tMean absolute percentage error  :",MAPE_Lum)
print(".......Accuracy for 80 m  ...", 100-MAPE_Lum)

plt.plot(testPredict_Lum, c = 'red', label = 'Predicted')
plt.plot(test_data_Lum , c = 'green', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Lumisity')
plt.title('Forecasted Lum')
plt.legend()
plt.savefig('Lum ARIMA ')
plt.show()


###################################

############# sAVING THE MODEL 
Name = ['MAE', 'MSE', 'RMSE', 'MAPE']  

# List2  
All_error_10 = [error_10, MAE_10, rmse_10, MAPE_10 ]  
All_error_45 = [error_45, MAE_45, rmse_45, MAPE_45]  
All_error_80 = [error_80, MAE_80, rmse_80, MAPE_80]
All_error_Temperature = [error_Temperature, MAE_Temperature, 
                         rmse_Temperature, MAPE_Temperature]
All_error_Humidity = [error_Humidity, MAE_Humidity, rmse_Humidity, MAPE_Humidity]
All_error_Pressure = [error_Pressure, MAE_Pressure, rmse_Pressure, MAPE_Pressure]
All_error_Luminisity = [error_Lum, MAE_Lum, rmse_Lum, MAPE_Lum]

Total_error_rate =     [All_error_10, All_error_45, All_error_80, All_error_Temperature , All_error_Humidity,  All_error_Pressure , All_error_Luminisity ]  

Total_error_rate_DF = pd.DataFrame(Total_error_rate, columns = Name) 


Total_error_rate_DF.rename(index={0:'Sm_10_cm', 
                                  1:'SM_45_cm', 
                                  2:'SM_80_cm',
                                  3:'Temperature',
                                  4:'Humidity',
                                  5:'Pressure',
                                  6:'Luminisity'}, inplace=True)

Total_error_rate_DF.to_csv("ARIMA Results.csv")