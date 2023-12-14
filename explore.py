import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import env
import os
import wrangle 

def plot_variable_pairs(df):
    sns.pairplot(train.sample(10000),kind='reg', hue='county', corner=True)
    plt.show()

def plot_categorical_and_continuous_vars(df, cat_vars, cont_vars):
    """
    Plots 3 different plots for visualizing a categorical variable and a continuous variable.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    cat_vars (list): The list of column names that hold the categorical features.
    cont_vars (list): The list of column names that hold the continuous features.
    """
    
    #for loop to output 3 different plots for each comparison
    for cat_var in cat_vars:
        for cont_var in cont_vars:
            # Plot a boxplot of the continuous variable for each category
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=cat_var, y=cont_var, data=df, palette="rocket")
            plt.title(f"{cont_var} by {cat_var}")
            plt.show()

            # Plot a histogram of the continuous variable for each category
            plt.figure(figsize=(12, 8))
            for cat in df[cat_var].unique():
                sns.histplot(
                    df[df[cat_var] == cat][cont_var],
                    label=cat,
                    alpha=0.5,
                    kde=True,
                    palette="rocket")
            plt.title(f"{cont_var} by {cat_var}")
            plt.legend()
            plt.show()

            # Plot a lineplot of the continuous variable for each category
            plt.figure(figsize=(12, 8))
            sns.lineplot(x=cat_var, y=cont_var, data=df, palette="rocket")
            plt.title(f"{cont_var} by {cat_var}")
            plt.show()

    
    #added decade and sq ft binned to better visualize the data
    #train["decade"] = pd.cut(train["yearbuilt"], bins=range(1880, 2021, 10), labels=range(1880, 2020, 10))
    #train["sq_feet_binned"] = pd.cut(train["square_feet"], bins=range(0, 13001, 1000), labels=range(0, 13000, 1000))

    #dropped nulls that were added from new columns
    #train = train.dropna()

    #changed datatypes to int
    #train["decade"] = train["decade"].astype(int)
    #train["sq_feet_binned"] = train["sq_feet_binned"].astype(int)
    
    #creating variables for my continuous and categorical features
    #cont = ["square_feet", "home_value", "tax"]
    #cat = ["decade", "county", "bathrooms", "bedrooms", "sq_feet_binned"]