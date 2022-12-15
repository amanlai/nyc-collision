# Predicting and Forecasting Serious Collisions in NYC

by Manlai Amarsaikhan

## Problem Statement

Through NYPD, the New York City makes datasets on motor vehicle collisions from 2012 on public via Open Data portal. In these datasets, the location, time, damage, number of injuries, deaths, vehicle type, etc. tens of features are collected.

(a) Given a variety of features including time, location and vehicle damage, can we predict whether a passenger was injured or died in a collision?

(b) Is it possible to forecast weekly total number of collisions where a passenger was injured or died?



## Data

### Data Collection

- Motor Vehicle Collisions - Crashes. This dataset contains records of collisions in NYC between January 2012 and December 2022.

- Motor Vehicle Collisions - Vehicles. This dataset contains records of vehicles in NYC that collided with another person/object.

- Motor Vehicle Collisions - Person. This dataset contains records of people who were present during a collision in NYC.

Given that they come from the same authority, these datasets are related. The relational schema between these datasets is as follows:

<img src="./images/schema.png">

- The collected datasets are not in this repository; however, the processed versions of them are placed in this repository's [data/processed_data](./data/processed_data/) folder.

### Data Cleaning

- The three datasets were merged on the common keys, namely, `Collision Id`, `Crash date` and `Crash time`.

- Since the datasets were pretty massive (each with over 1.7 million observations), only the data for Manhattan was used.

- Given that a collision may include multiple vehicles, there were multiple columns of similar data such as Contributing Factor, Vehicle Type, Vehicle Damage where each column corresponded to a different vehicle. Some of these columns were missing a lot values. These columns were combined into a single column and the rest were dropped.

- Certain columns such as `Public Property Damage` were missing >95% observations, so these were dropped as well.

- Since location is an important feature to explore in this study, any observations missing Latitude and Longitude data were dropped.

- After the above procedure, the remaining dataframe was of shape `(558724, 51)`. It was necessary to drop redundant columns and it was necessary to perform EDA to determine which columns were surplus and which were not.


---


### Exploratory Data Analysis and Feature Engineering

- First of all, the target variable of the classification problem was constructed. The dataset contained the following for each observation:

    - Number of persons injured
    - Number of persons killed
    - Number of pedestrians injured
    - Number of pedestrians killed
    - Number of cyclists injured
    - Number of cyclists killed
    - Number of motorists injured
    - Number of motorists killed

  All of them used to create a binary `Serious Collision` column. In other words, a collision was deemed _"serious"_ if any person was injured or killed as a result of it.

  The aim of the first half of the project is to predict these collisions.

- There were many columns that explain the severity of a collision. Examples include `Contributing Factor`, `Vehicle Damage`, `Vehicle Type`, `State Registration`, `Driver License` etc. All of them were explored to see if there were any significant correlation between them and `Serious Collision`.

<p>
<img src="./images/eda_damage.png", width=40%>
<img src="./images/eda_traffic.png", width=40%>
</p>


<img src="./images/clusters.png">
<img src="./images/clusters_2.png">


<img src="./images/eda_vehicle.png">
<img src="./images/eda_night.png">




<img src="./images/weekly_collisions.png">
<img src="./images/seasonal_decompose.png">
<img src="./images/pacf.png">

---

## Classifying Serious Collisions

### First Pass

<img src="./images/first_pass.png">

The notebook of this model is placed in this repository's [code/](./code/4-first-pass.ipynb) folder.


### Second Try

<img src="./images/second_pass.png">

The notebook of this set of models is placed in this repository's [code/](./code/5-second-model.ipynb) folder.

### Third Pass: 

<img src="./images/third_pass.png">

The notebook of this set of models is placed in this repository's [code/](./code/6-third-model.ipynb) folder.

### Fourth Pass: Over/Under Sampling the Minority Class

<img src="./images/fourth_pass.png">

The notebook of this set of models is placed in this repository's [code/](./code/7-over-under-sampled.ipynb) folder.

---

## Forecasting Weekly Total Number of Serious Collisions

### First Model: OLS

<img src="./images/weekly_collisions_forecast.png">

The notebook of this model is placed in this repository's [code/](./code/8-time-series-ols.ipynb) folder.

### Second Model: LSTM

<img src="./images/weekly_collisions_forecast_lstm.png">

<img src="./images/lstm_loss.png">
<img src="./images/lstm_r2.png">



The notebook of this model is placed in this repository's [code/](./code/10-lstm.ipynb) folder.









---

## Conclusion

