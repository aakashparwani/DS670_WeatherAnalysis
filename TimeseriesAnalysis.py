# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:53:25 2017

@author: hp
"""

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

from numpy.linalg import LinAlgError

#Let's update null values in our dataset. 
dewpoint = updatenullvalues(dewpoint)    
humidity = updatenullvalues(humidity)    
pressure = updatenullvalues(pressure)    
temperature = updatenullvalues(temperature)    
winddirection = updatenullvalues(winddirection)    
windspeed = updatenullvalues(windspeed)    

#weather_dataset = pd.concat([dewpoint,humidity.ix[:,1],pressure.ix[:,1],temperature.ix[:,1],winddirection.ix[:,1],windspeed.ix[:,1]],axis=1)

dewpoint_feb_jun1 = pd.read_csv(inputPath+"/dewpoint/dewptm_Feb_Jun.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')


dewpoint_aug_sep1 = pd.read_csv(inputPath+"/dewpoint/dewptm_Aug_Sep.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')

humidity_feb_jun1 = pd.read_csv(inputPath+"/humidity/hum_feb_jun.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')
humidity_aug_sep1 = pd.read_csv(inputPath+"/humidity/hum_aug_sep.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')

pressure_feb_jun1 = pd.read_csv(inputPath+"/pressure/pressurem_feb_jun.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')
pressure_aug_sep1 = pd.read_csv(inputPath+"/pressure/pressurem_aug_sept.csv"
                                ,parse_dates=['DateTime'], index_col='DateTime')

temp_feb_jun1 = pd.read_csv(inputPath+"/temperature/tempm_feb_jun.csv"
                            ,parse_dates=['DateTime'], index_col='DateTime')
temp_aug_sep1 = pd.read_csv(inputPath+"/temperature/tempm_aug_sept.csv"
                            ,parse_dates=['DateTime'], index_col='DateTime')

winddirection_feb_jun1 = pd.read_csv(inputPath+"/winddirection/wdird_feb_jun.csv"
                                     ,parse_dates=['DateTime'], index_col='DateTime')
winddirection_aug_sep1 = pd.read_csv(inputPath+"/winddirection/wdird_aug_sept.csv"
                                     ,parse_dates=['DateTime'], index_col='DateTime')

windspeed_feb_jun1 = pd.read_csv(inputPath+"/windspeed/wspdm_feb_jun.csv"
                                 ,parse_dates=['DateTime'], index_col='DateTime')
windspeed_aug_sep1 = pd.read_csv(inputPath+"/windspeed/wspdm_aug_sept.csv"
                                 ,parse_dates=['DateTime'], index_col='DateTime')


##concatinate the data, first row wise.
dewpoint = pd.concat([dewpoint_feb_jun1,dewpoint_aug_sep1])
humidity = pd.concat([humidity_feb_jun1,humidity_aug_sep1])
pressure = pd.concat([pressure_feb_jun1,pressure_aug_sep1])
temperature = pd.concat([temp_feb_jun1,temp_aug_sep1])
winddirection = pd.concat([winddirection_feb_jun1,winddirection_aug_sep1])
windspeed = pd.concat([windspeed_feb_jun1,windspeed_aug_sep1])

def updatenullvalues(dataset):
    for col in dataset.ix[:,1:]:
        if dataset[col].isnull().any:
            mean = dataset[col].mean()
            dataset[col].fillna(mean,inplace=True)
    return dataset 
    

for col in dataset.ix[:,1:]:
        if dataset[col].isnull().any:
            mean = dataset[col].mean()
            dataset[col].fillna(mean,inplace=True)
return dataset    
#Let's update null values in our dataset. 
dewpoint = updatenullvalues(dewpoint)    
humidity = updatenullvalues(humidity)    
pressure = updatenullvalues(pressure)    
temperature = updatenullvalues(temperature)    
winddirection = updatenullvalues(winddirection)    
windspeed = updatenullvalues(windspeed)    


data = temperature

print (data.head())

data.index

ts = data['Temperature']

plt.plot(ts)




from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)    
test_stationarity(ts)    

ts_log = np.log(ts)
plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

#make rolling mean stationary
moving_avg = pd.rolling_mean(ts_log,12)
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


ts_log = ts_log[~np.isinf(ts_log)]
ts_log = ts_log[~np.isnan(ts_log)]


ts_log_diff = ts_log_diff[~np.isinf(ts_log_diff)]
ts_log_diff = ts_log_diff[~np.isnan(ts_log_diff)]                
                
nancount = 0
infcount = 0
for i in temperature['Temperature']:
    if np.isnan(i) == True:
        nancount = nancount + 1

for i in temperature['Temperature']:    
    if np.isinf(i) == True:
        infcount = infcount + 1        

        

nancount = 0
infcount = 0
for i in ts_log:
    if np.isnan(i) == True:
        nancount = nancount + 1
        

for i in ts_log:
    if np.isinf(i) == True:
        infcount = infcount + 1
        
nancount        
infcount

np.mean(ts_log)

ts_log_new = pd.DataFrame(ts_log)





#######AR Model

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


#######MA Model
model = ARIMA(ts_log, order=(0, 1, 1))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


######ARIMA Model
model = ARIMA(ts_log, order=(1, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('ARIMA model (2,1,1)')



predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())


predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))