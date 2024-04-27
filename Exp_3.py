import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np 
from scipy import stats
from scipy.stats import norm, probplot
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score

#Visualization Libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train = pd.read_csv('/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/input/house-prices-advanced-regression-techniques/test.csv')
train.shape
test.shape
train.info()
cat_cols = train.select_dtypes(include=['object', 'O']).columns
num_cols = train.select_dtypes(include=['int64', 'float64'])
for feature in num_cols:
    zero_values = (train[feature] == 0).sum()
    null_values = train[feature].isnull().sum()
    unique_values = len(train[feature].unique())
    print(f"Feature: {feature}")
    print(f"Number of 0 Values: {zero_values}")
    print(f"Number of Null Values: {null_values}")
    print(f"Unique Values: {unique_values}")
    print("="*30)
cat_cols = train.select_dtypes(include=['object', 'O']).columns
num_cols = train.select_dtypes(include=['int64', 'float64'])

for feature in num_cols:
    zero_values = (train[feature] == 0).sum()
    null_values = train[feature].isnull().sum()
    unique_values = len(train[feature].unique())
    print(f"Feature: {feature}")
    print(f"Number of 0 Values: {zero_values}")
    print(f"Number of Null Values: {null_values}")
    print(f"Unique Values: {unique_values}")
    print("="*30)
cat_cols = train.select_dtypes(include=['object', 'O']).columns

num_cols = train.select_dtypes(include=['int64', 'float64'])

for feature in num_cols:
    zero_values = (train[feature] == 0).sum()
    null_values = train[feature].isnull().sum()
    unique_values = len(train[feature].unique())

    print(f"Feature: {feature}")
    print(f"Number of 0 Values: {zero_values}")
    print(f"Number of Null Values: {null_values}")
    print(f"Unique Values: {unique_values}")
    print("="*30)
test["PoolQC"] = test["PoolQC"].fillna("NA")
test["MiscFeature"] = test["MiscFeature"].fillna("NA")
test["Alley"] = test["Alley"].fillna("NA")
test["Fence"] = test["Fence"].fillna("NA")
test["FireplaceQu"] = test["FireplaceQu"].fillna("NA")
test["MasVnrType"] = test["MasVnrType"].fillna("NA")
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
test["Functional"] = test["Functional"].fillna("Typ")


test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

train['MSSubClass'] = train['MSSubClass'].fillna("NA")

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    test[col] = test[col].fillna('NA')
# Since there is no garage, I fill the null values with 0.
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    test[col] = test[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    test[col] = test[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test[col] = test[col].fillna('NA')
plt.figure(figsize = (8, 4), dpi = 100)
sns.boxplot(x = "Neighborhood", y = "LotFrontage", data = train)
plt.xticks(rotation = 90)
plt.show()
train.drop([523, 1298, 1337, 934, 297, 440, 185, 495],axis=0,inplace=True)
train['totalsf'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['BsmtFinSF1'] + train['BsmtFinSF2']
test['totalsf'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['BsmtFinSF1'] + test['BsmtFinSF2']
train['totalarea'] = train['GrLivArea'] + train['TotalBsmtSF']
test['totalarea'] = test['GrLivArea'] + test['TotalBsmtSF']
# We save the line numbers. We will split the data we will merge later back into train and test.
ntrain = train.shape[0]

# Target Variable
y_train = train.SalePrice.values

# Merges the training and test datasets. Resets the indexes.
all_data = pd.concat((train, test)).reset_index(drop=True)

# Drops the 'SalePrice' column.
all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is: {}".format(all_data.shape))
ccol = all_data.dtypes[train.dtypes == "object"].indexccol
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
# We applied a logarithmic transformation. We need to convert it back to its original scale and predict the test accordingly.
test_pred = np.expm1(model_xgb.predict(test))
len(test_pred)

