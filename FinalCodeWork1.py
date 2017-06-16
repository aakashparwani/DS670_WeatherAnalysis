# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:28:55 2017

@author: hp
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn import preprocessing

from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split

from sklearn import linear_model

from sklearn.metrics import classification_report

# Importing train_test_split that will quickly split data into train and test
from sklearn.cross_validation import train_test_split

# Importing PCA from sklearn.decomposition to perform pca functionality
from sklearn.decomposition import PCA

import matplotlib.pyplot as plot
# Importing Confusion_matrix from sklearn,metrics to create confusion matrix
from sklearn.metrics import confusion_matrix

# model_selection.cross_val_score takes a scoring parameter that controls
# what metric they apply to the estimators evaluated
from sklearn import model_selection

# Importing accuracy from sklearn.metrics to calculate accuracy score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC


sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

inputPath =  "D:/Aakash_Documents/MS_Collections/AcceptanceFromSaintPeters/ClassStuff/DS_670_Capstone/FinalProject_WeatherReport/dataset/raw_weather_data_aarhus"

dewpoint_feb_jun1 = pd.read_csv(inputPath+"/dewpoint/dewptm_Feb_Jun.csv")
dewpoint_aug_sep1 = pd.read_csv(inputPath+"/dewpoint/dewptm_Aug_Sep.csv")

humidity_feb_jun1 = pd.read_csv(inputPath+"/humidity/hum_feb_jun.csv")
humidity_aug_sep1 = pd.read_csv(inputPath+"/humidity/hum_aug_sep.csv")

pressure_feb_jun1 = pd.read_csv(inputPath+"/pressure/pressurem_feb_jun.csv")
pressure_aug_sep1 = pd.read_csv(inputPath+"/pressure/pressurem_aug_sept.csv")

temp_feb_jun1 = pd.read_csv(inputPath+"/temperature/tempm_feb_jun.csv")
temp_aug_sep1 = pd.read_csv(inputPath+"/temperature/tempm_aug_sept.csv")

winddirection_feb_jun1 = pd.read_csv(inputPath+"/winddirection/wdird_feb_jun.csv")
winddirection_aug_sep1 = pd.read_csv(inputPath+"/winddirection/wdird_aug_sept.csv")

windspeed_feb_jun1 = pd.read_csv(inputPath+"/windspeed/wspdm_feb_jun.csv")
windspeed_aug_sep1 = pd.read_csv(inputPath+"/windspeed/wspdm_aug_sept.csv")


##concatinate the data, first row wise.
dewpoint = pd.concat([dewpoint_feb_jun1,dewpoint_aug_sep1])
humidity = pd.concat([humidity_feb_jun1,humidity_aug_sep1])
pressure = pd.concat([pressure_feb_jun1,pressure_aug_sep1])
temperature = pd.concat([temp_feb_jun1,temp_aug_sep1])
winddirection = pd.concat([winddirection_feb_jun1,winddirection_aug_sep1])
windspeed = pd.concat([windspeed_feb_jun1,windspeed_aug_sep1])

#In this step, we will try to update null values.
#filling null values could be complicated.As we seen in previous data exploration steps
#that 116 was the maximum null values and total datasize is 12563. Since, maximum percent of null values is less than 10.
#So, null values will be replaced by mean of the particular parameter.
def updatenullvalues(dataset):
    for col in dataset.ix[:,1:]:
        if dataset[col].isnull().any or dataset[col].isinf().any or dataset[col].isnan().any:
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

weather_dataset = pd.concat([dewpoint,humidity.ix[:,1],pressure.ix[:,1],temperature.ix[:,1],winddirection.ix[:,1],windspeed.ix[:,1]],axis=1)

ss.kstest(dewpoint['DewPoint'], ss.randint.cdf, args=(0,10))

ss.kstest(X_normalized, ss.randint.cdf, args=(0,10))

ss.kstest(humidity['Humidity'], ss.randint.cdf, args=(0,10))
ss.kstest(temperature['Temperature'], ss.randint.cdf, args=(0,10))
ss.kstest(pressure['Pressure'], ss.randint.cdf, args=(0,10))

ss.kstest(winddirection['WindDirection'], ss.randint.cdf, args=(0,10))
ss.kstest(windspeed['WindSpeed'], ss.randint.cdf, args=(0,10))

X_normalized = preprocessing.normalize(dewpoint['DewPoint'], norm='l2')

X_normalized = preprocessing.normalize(dewpoint['DewPoint'], norm='l2')

X_normalized.reshape(-1,1)

sns.distplot(X_normalized)

sns.distplot(weather_dataset['DewPoint'])
sns.distplot(weather_dataset['Humidity'])
sns.distplot(weather_dataset['Temperature'])
sns.distplot(weather_dataset['Pressure'])

sns.distplot(weather_dataset['WindSpeed'])

weather_dataset.describe()

sns.regplot(humidity['Humidity'], temperature['Temperature'])

sns.regplot(windspeed['WindSpeed'], temperature['Temperature'])

sns.lmplot(x="Humidity", y="Temperature", data=weather_dataset,
           y_jitter=.03);

########good           
sns.lmplot(x="DewPoint", y="Temperature", data=weather_dataset,
           y_jitter=.03);

           
sns.lmplot(x="Pressure", y="Temperature", data=weather_dataset,
           y_jitter=.03);           

pearsonr(weather_dataset['DewPoint'], weather_dataset['Temperature'])

print("Pearson's correlation coefficient, between dewpoint & windspeed",
      pearsonr(weather_dataset['DewPoint'],weather_dataset['Temperature'])[0])

print("P-Value is",pearsonr(weather_dataset['DewPoint'], 
                            weather_dataset['Temperature'])[1])           

pearsonr(weather_dataset['Humidity'], weather_dataset['Temperature'])

pearsonr(weather_dataset['DewPoint'], weather_dataset['Humidity'])

pearsonr(np.log(weather_dataset['Pressure']), weather_dataset['Temperature'])

pearsonr(np.log(weather_dataset['WindSpeed']), weather_dataset['Temperature'])


weather_dataset[weather_dataset['WindSpeed']>0].count()

weather_dataset[weather_dataset['WindSpeed']<0].count()

weather_dataset[weather_dataset['WindSpeed']==0].count()

weather_dataset[weather_dataset['WindSpeed']==0]

fig, axis = plt.subplots()
axis.set_title("Relation Temperature & DewPoint")
axis.set_xlabel('Pressure')
axis.set_ylabel('WindSpeed')

plt.plot(weather_dataset['Pressure'], weather_dataset['WindSpeed'])
plt.show()


weather_dataset = pd.concat([dewpoint,humidity.ix[:,1],pressure.ix[:,1],temperature.ix[:,1],winddirection.ix[:,1],windspeed.ix[:,1]],axis=1)

weather_dataset_new = pd.concat([dewpoint.ix[:,1],humidity.ix[:,1],pressure.ix[:,1]],axis=1)

def dividedata(train_data,target_data):
    train,test,train_target,test_target = train_test_split(train_data,target_data,test_size=0.20,random_state=42)
    return train,test,train_target,test_target
    
train,test,train_y,test_y = dividedata(weather_dataset_new,temperature.ix[:,1])


seed = 7 # 7 is just the id, can be any number
scoring = 'accuracy'

# Checking Algorithms
# Creating empty list to use it for every model in for loop
models = []

models.append(('LR', LogisticRegression()))
models.append(('PR', Perceptron()))

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))
#models.append(('RNC', RadiusNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

# evaluating each model in turn
results = []
names = []
msg=[]


# Creating for loop to call different models
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #cv_results = model_selection.cross_val_score(model, train_data, target_data,cv = 5,scoring="f1") 
    cv_results = model_selection.cross_val_score(model, train, train_y,cv = 5,scoring="f1") 
    results.append(cv_results)
    names.append(name)
    msg.append ("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))


regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(train, train_y)

result= regr.fit(train, train_y)

train

# The coefficients
print('Coefficients: \n', regr.coef_)


# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(test) - test_y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test, test_y))

print(result.summary())

test.count()
test_y.count()
# Plot outputs
plt.scatter(test, test_y,  color='black')

plt.plot(test_y, regr.predict(test), color='blue',
         linewidth=1)

plt.scatter(test_y, regr.predict(test), color='blue')


plt.xticks(())
plt.yticks(())

plt.show()    

###################
sns.pairplot(weather_dataset, vars=['DewPoint', 'Humidity', 'Pressure'],kind='reg')

###############NLP
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

newtrain_y = pd.DataFrame(train_y)
clf.fit(train[["DewPoint","Humidity"]], newtrain_y[["Temperature"]].astype(int)) 

clf.predict(test)

plt.scatter(test_y, clf.predict(test), color='blue')

plt.plot(test_y, clf.predict(test), color='blue')

plt.scatter(test, test_y,  color='black')

##############

import statsmodels.formula.api as smf

# create a fitted model with all three features
lm = smf.ols(formula='Temperature ~ DewPoint + Humidity + Pressure', data=weather_dataset).fit()

# print the coefficients
lm.params

lm.summary()

lm.rsquared


# create a fitted model with all three features
lm = smf.ols(formula='Temperature ~ DewPoint + Humidity', data=weather_dataset).fit()

# print the coefficients
lm.params

lm.summary()

lm.rsquared

fig, axis = plt.subplots()
axis.set_title("Comparison True value & Predicted Value")
axis.set_xlabel('True value of temperature')
axis.set_ylabel('Predicted value of temperature')


plt.plot(test_y, lm.predict(test), color='blue',
         linewidth=1)



###################
sns.pairplot(weather_dataset, vars=['Temperature','DewPoint', 'Humidity', 'Pressure'],kind='reg')