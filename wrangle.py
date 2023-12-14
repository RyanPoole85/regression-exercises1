import pandas as pd
import numpy as np

import env
import os
from sklearn.model_selection import train_test_split

def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function takes in 1 positional arguement and checks for username, host, and password credentials from imported env module. 
    Returns a formatted connection url to access mySQL database.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    

def check_file_exists(filename, query, url):
    '''
    checks if file already exists.
    '''
    if os.path.exists(filename):
        print('this file exists, reading csv')
        df = pd.read_csv(filename, index_col=0)
    else:
        print('this file doesnt exist, read from sql, and export to csv')
        df = pd.read_sql(query, url)
        df.to_csv(filename)
        
    return df


def get_zillow_data():
    '''
    function pulls zillow data from the MySQL Codeup db into a dataframe.
    '''
    url = env.get_db_url('zillow')
    query = '''
    select bedroomcnt, bathroomcnt,calculatedfinishedsquarefeet, taxvaluedollarcnt,
	yearbuilt, taxamount, fips from properties_2017
    join propertylandusetype using (propertylandusetypeid)
    where propertylandusedesc in ('Single Family Residential');
    '''
    
    filename = 'zillow.csv'

    #call the check_file_exists fuction 
    df = check_file_exists(filename, query, url)
    return df

def prep_zillow(df):
    
    #drop all nulls
    df = df.dropna()
    
    #change datatype of  exam1 and exam3 to integers
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.yearbuilt = df.yearbuilt.astype(int)
    df.fips = df.fips.astype(int)
    
    #rename columns
    df = df.rename(columns={df.columns[0]: 'bedrooms', df.columns[1]: 'bathrooms', df.columns[2]: 'square_feet',\
                   df.columns[3]: 'home_value', df.columns[5]: 'tax', df.columns[6]: 'county'})
    
    df.county = df.county.map({6037:'LA', 6059:'Orange', 6111:'Ventura'})
    
    return df

def splitting_data(df):
    '''
    Takes in a df and a column (target variable) and splits into df, validate and test. 
    Ex: df, validate, test = prepare_telco.splitting_data(df, 'churn')
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=123
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123
                        )
    return train, validate, test


def wrangle_zillow():
    train, validate, test=splitting_data(prep_zillow(get_zillow_data()))    
    return train, validate, test