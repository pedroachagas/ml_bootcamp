import pandas as pd
import numpy as np

from IPython.core.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# parameters
n_splits = 5
C = 1.0
output_file = f'week_5/model_C{C}.bin'

# dependent variables
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender','seniorcitizen','partner','dependents',
'phoneservice','multiplelines','internetservice','onlinesecurity',
'onlinebackup','deviceprotection','techsupport','streamingtv',
'streamingmovies','contract','paperlessbilling','paymentmethod']

# data preparation
def preprocess(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']

    return df_full_train, df_train, df_val, df_test, y_train, y_val, y_test

def vectorize(df):
    global numerical
    global categorical
    
    dv = DictVectorizer(sparse=False)

    dict = df[categorical + numerical].to_dict(orient='records')
    X = dv.fit_transform(dict)

    return X, dv

def train_model(df_train, y_train, C=1):
    X_train, dv = vectorize(df_train)
    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return model, dv

def predict_model(df, dv, model):
    global numerical
    global categorical
        
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = preprocess('week_5/telco_churn.csv')

# validation
print(f'\nDoing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    # the k-fold split uses index to shuffle the data
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    # y values come from dataset
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    # training and predicting
    model, dv = train_model(df_train, y_train, C=C)
    y_pred = predict_model(df_val, dv, model)
    
    # AUC
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print (f'auc on fold {fold} is {auc}')
    fold += 1

print('\nvalidation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

#training the final model
print('\ntraining the final model:')

model, dv = train_model(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict_model(df_test, dv, model)

y = df_test['churn'].values
auc = roc_auc_score(y_test, y_pred)
print('auc =', auc)

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'\nModel is saved to {output_file}')

