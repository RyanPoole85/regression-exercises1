import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

# KNN, a distance based classifier
from sklearn.neighbors import KNeighborsClassifier
# scaling objects:
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import env
import os
import explore
import wrangle 

def prepare_zillow(train, validate, test):
    """takes in train, validate, test zillow data and prepares it for modeling. Returns train, validate, test. """
    #encode for modeling
    train = pd.concat([train, pd.get_dummies(train["county"])], axis=1)
    validate = pd.concat([validate, pd.get_dummies(validate["county"])], axis=1)
    test = pd.concat([test, pd.get_dummies(test["county"])], axis=1)

    # Drop the original 'county' column
    train = train.drop("county", axis=1)
    validate = validate.drop("county", axis=1)
    test = test.drop("county", axis=1)

    #rename for modeling
    train.rename(columns={"LA": "los_angeles", "Orange": "orange", "Ventura": "ventura"},
        inplace=True)

    validate.rename(columns={"LA": "los_angeles", "Orange": "orange", "Ventura": "ventura"},
        inplace=True)

    test.rename(columns={"LA": "los_angeles", "Orange": "orange", "Ventura": "ventura"},
        inplace=True)

    return train, validate, test

def qt_scaler(train, validate, test): 
    """Takes in train, validate, test and prepares them for scaling for quantile transformer w/normal output distribution. 
    returns X_train_scaled, X_validate_scaled, X_test_scaled and plots original and scaled data."""
    
    #set X and Y variables
    X_train, X_validate, X_test = train[['square_feet', 'bathrooms','yearbuilt', 'los_angeles','orange','ventura']],\
                                  validate[['square_feet','bathrooms', 'yearbuilt', 'los_angeles', 'orange','ventura']],\
                                  test[['square_feet','bathrooms', 'yearbuilt', 'los_angeles', 'orange','ventura']]
    y_train, y_validate, y_test = train.home_value, validate.home_value, test.home_value


    #quantile transformer with normal output distribution

    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    #set scaled variables
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    #plot it
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, bins=25, ec='black')
    plt.title('Scaled');
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

#must set columns to scale
#example: columns_to_scale = ['bedrooms', 'bathrooms','taxamount', 'area']
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()

#examples of calling above
# MinMaxScaler Applied
# visualize_scaler(scaler=MinMaxScaler(), 
#                df=train, 
#                columns_to_scale=columns_to_scale, 
#                bins=50)

# StandardScaler Applied
# visualize_scaler(scaler=StandardScaler(), df=train, columns_to_scale=columns_to_scale, bins=50)

# RobustScaler Applied
#visualize_scaler(scaler=RobustScaler(), df=train, columns_to_scale=columns_to_scale, bins=50)

# QuantileTransformer Applied
#visualize_scaler(scaler=QuantileTransformer(output_distribution='normal'), 
#                 df=train,
#                 columns_to_scale=columns_to_scale, 
#                 bins=50)

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_amount', 'sq_feet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    #example of calling above
    #scaler, train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, return_scaler=True)

