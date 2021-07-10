import pandas as pd

df=pd.read_csv('car data.csv')

#print(df.head()) 
#below code segment for checking data
print("*************************************************")

print(df.shape)

print("Unique Seller Types: "+df['Seller_Type'].unique())
print("Unique Fuel Types: "+df['Fuel_Type'].unique())
print("Unique Transmisson Types: "+df['Transmission'].unique())

print(df.isnull().sum())

print("*************************************************")

#create newtable with new columns
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset['Current Year']=2021 #create new col with current year
final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year'] #create new col for car number of years upto date used 
#print(final_dataset)
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current Year'],axis=1,inplace=True)
#print(final_dataset.head())

final_dataset=pd.get_dummies(final_dataset,drop_first=True) # drop first will remove cng col as onehot encoding, if we know
                                                            # col1=0, col2=0, so col3 will be one, thus one col less is required
#final_dataset.head()

print(final_dataset.corr()) # this shows how the features are correlated to each other

import seaborn as sns 
import matplotlib.pyplot as plt
sns.pairplot(final_dataset)
#plt.show()

corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()

X=final_dataset.iloc[:,1:] # features
y=final_dataset.iloc[:,0]  # labels

#print(X.head())
#print(y.head())

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_) # shows all columns correlation

# Now model training starts

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV # for choosing best parameters for hyper tuning

 #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)

print(rf_random.best_params_)
print(rf_random.best_score_)

predictions=rf_random.predict(X_test)
sns.displot(y_test-predictions)
plt.scatter(y_test,predictions)
#plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("^^^^^^^^^^^^^")
#to save the model
import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)