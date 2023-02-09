from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



'''class general1:
    def __init__(self) -> None:
        pass'''
# encoding
# method{‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
# axis=0, 1
def fillna0(data):
    data = data.fillna(0) # fillna
    return data

def fillnabackfill(data, axis = 0):
    data = data.fillna(method='backfill', axis = axis) # fillna
    return data

def fillnabfill(data, axis = 0):
    data = data.fillna(method='bfill', axis = axis) # fillna
    return data

def fillnapad(data, axis = 0):
    data = data.fillna(method='pad', axis = axis) # fillna
    return data

def fillnaffill(data, axis = 0):
    data = data.fillna(method='ffill', axis = axis) # fillna
    return data

def fillnamean(data):
    for i in data:
        if len(data[i][data[i].isnull() == True]) != 0:
            data[i] = data[i].fillna(value=data[i].mean()) # fillna
    return data


# encoding
def drop_isable(data):
    data = data[data.ISABLE != 1] # drop row with isable == 1
    return data

# encoding
def drop_repeatcolumn(data):
    data = data.drop(['AGENT_ID'], axis = 1)
    data = data.drop(['ISABLE'], axis = 1)
    data = data.drop(['END_DATE'], axis = 1)
    data = data.drop(['BIR_DATE'], axis = 1)
    data = data.drop(['DISTRICT'], axis = 1)
    return data

# encoding
def separate_Xy(data):
    data_X = data.drop(['life_insurance'], axis = 1)
    data_X = data_X.drop(['property_insurance'], axis = 1)
    data_y = data[['life_insurance', 'property_insurance']]
    return data_X, data_y

# encoding
def encoding_onehot(data):    
    categorical = []
    for i in data:        
        if type(data[f"{i}"][1]) == str:
            categorical.append(i)
    for i in categorical:
        for z, k in enumerate(data[i]):
            if type(k) != str:
                data[i].iloc[z] = str(data[i].iloc[z])         
        dummies = pd.get_dummies(data[f"{i}"], prefix = i)
        data = pd.concat([data, dummies], axis = 1)
        data = data.drop([i], axis = 1) 
            #encoder.fit(np.array(data[f"{i}"]).reshape(-1, 1))
            #encoder.transform(data[f"{i}"])
    return data

# encoding
def outlier_Isolation(X, y):
    X_d = X
        
    for i in X_d:        
            try:
                X_d[i] = X_d[i].astype(float)            
                print(i)
            except:
                X_d = X_d.drop([i], axis = 1)
        
    isolation_forest = IsolationForest(random_state=42)
    outlier_pred = isolation_forest.fit_predict(X_d)
    X = X.iloc[outlier_pred == 1]
    y = y.iloc[outlier_pred == 1]
    return outlier_pred, X, y

# train_test_split
def outlier_std(X, y, t): # t: times the std
    X_d = X    
    for i in X_d:        
        try:
            X_d[i] = X_d[i].astype(float)    
        except:
            X_d = X_d.drop([i], axis = 1)

    outlier_pred = np.ones(len(X)).reshape(-1, 1)
    
    for i in X_d:
        upperbound = X_d[i].mean() + X_d[i].std()*t 
        lowerbound = X_d[i].mean() - X_d[i].std()*t  
        for c, k in enumerate(X_d[i]):
            if k>upperbound or k<lowerbound:
                outlier_pred[c] = -1
                
    X = X.iloc[outlier_pred == 1]
    y = y.iloc[outlier_pred == 1]
    
    return outlier_pred, X, y

# train_test_split
def date2months(datevalue, base_year):
    #year = [int(i/10000) for i in datevalue]
    #month = [int((i-int(i/10000)*10000)/100) for i in datevalue]
    months = [(int(i/10000) - base_year)*12 + int((i-int(i/10000)*10000)/100)     
        for i in datevalue]
    return months

# train_test_split
def split_traintest(data, testsize):    
    train, test = train_test_split(data, test_size = testsize)
    return train, test




