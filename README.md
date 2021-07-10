## Screen shots:

![imgonline-com-ua-resize-slU16k7PvFd](https://user-images.githubusercontent.com/55889161/125172695-c63cf000-e1d8-11eb-9dc5-cb355f80b800.jpg)
![imgonline-com-ua-resize-zGFZR92ISU](https://user-images.githubusercontent.com/55889161/125172697-c63cf000-e1d8-11eb-8f66-6eed7cdb0c6d.jpg)
![imgonline-com-ua-resize-ssGGZcfjWucp8](https://user-images.githubusercontent.com/55889161/125172693-c50bc300-e1d8-11eb-875e-e031e90ea59a.jpg)

## About the code:

The data has Car name, type of transmission, fuel type,... all such parameters to determine the price of car for current year (2021 in this case).
The dataset is available on kaggle.

https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho 

The **main.py** file contains the main code for model traing.
The **app.py** file has the code required for sending request to backend for display results.
It is this file which the heroku calls.

We have used Linear Regression for this purpose.
The main Purpose of this project is to revise **linear regression** and deployment of project on **heroku.**

Points:
 - Check the correlation between features using displot and scatter
 - Use of RandomForestRegressor() (no feature scaling required as has decision tree)
 - Metrics used MAE,MSE and RMSE

The trained model will be generated as **random_forest_regression_model.pkl**




