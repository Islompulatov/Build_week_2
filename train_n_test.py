import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics, pipeline
import time
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer




df = pd.read_csv("./dataset_5secondWindow/dataset_5secondWindow.csv")
df = df.dropna(axis=1, how="any", thresh=len(df)*.5, subset=None, inplace=False)
not_null = [col for col in df.columns if df[col].isnull().sum() < 1]
new_null = [i for i in df.columns if 1<= df[i].isnull().sum()<2374]

df1 = df[new_null].rolling(window=10, min_periods=1).mean()

df2= df[not_null]
df3= pd.concat([df2,df1], axis = 1)
df3 =df3.dropna(axis=0, how="any")

df4 = df.sort_values(by='user', ascending=True)

OE = OrdinalEncoder()
ct2 = np.asarray(df3['user'])
df3['user'] = OE.fit_transform(ct2.reshape(-1,1))
df3 = df3.sort_values(by='user', ascending=False)


new_df = df3.copy()
ct = np.asarray(new_df['target'])
new_df['target'] = OE.fit_transform(ct.reshape(-1,1))



df_train = new_df.iloc[:3421,:]
df_test = new_df.iloc[3421:,:]


X_train = df_train.drop(['target','user'],axis=1)
#removing id column and unnamed column. not a feature needed for the training
X_train = X_train.iloc[:,2:]
y_train = df_train['target']
#print(X_train)

# Test set 
x_test = df_test.drop(['target','user'],axis =1)
x_test= x_test.iloc[:,2:]
y_test = df_test['target']


#traning on raw data after replacing the missing value ,without feature engineer
tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":ExtraTreesClassifier(),
  "Random Forest":RandomForestClassifier(),
  "AdaBoost":AdaBoostClassifier(),
  "Skl GBM": GradientBoostingClassifier(),
  "Skl HistGBM":HistGradientBoostingClassifier(),
  "XGBoost": XGBClassifier(),
  "LightGBM":LGBMClassifier(),
  "CatBoost": CatBoostClassifier(),
  "SVM":      SVC()}
tree_classifiers = {name: pipeline.make_pipeline( model) for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              
                              
results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
print(results_ord)



#feature engineer, classifying the target into two
new_df1 = df3.copy()
new_df1['target']= new_df1['target'].apply({'Bus':0, 'Car':0,  'Train':0, 'Still':1,'Walking':2}.get) 
#spliting the data by users
df_train1 = new_df1.iloc[:3421,:]
df_test1 = new_df1.iloc[3421:,:]


X_train1 = df_train1.drop(['target','user'],axis=1)
#removing id column and unnamed column. not a feature needed for the training
X_train1 = X_train1.iloc[:,2:]
y_train1 = df_train1['target']
#print(X_train)
# Test set 

x_test1 = df_test1.drop(['target','user'],axis =1)
x_test1= x_test1.iloc[:,2:]
y_test1 = df_test1['target']


#traning on data that target feature has been altered by classifying the target into 3 classes 
tree_classifiers1 = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":ExtraTreesClassifier(),
  "Random Forest":RandomForestClassifier(),
  "AdaBoost":AdaBoostClassifier(),
  "Skl GBM": GradientBoostingClassifier(),
  "Skl HistGBM":HistGradientBoostingClassifier(),
  "XGBoost": XGBClassifier(),
  "LightGBM":LGBMClassifier(),
  "CatBoost": CatBoostClassifier(),
  "SVM":      SVC()}
tree_classifiers1 = {name: pipeline.make_pipeline( model) for name, model in tree_classifiers1.items()}

results1 = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in tree_classifiers1.items():
    
    start_time = time.time()
    model.fit(X_train1, y_train1)
    total_time = time.time() - start_time
        
    pred1 = model.predict(x_test1)
    
    results1 = results1.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test1, pred1)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test1, pred1)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              
                              
results_ord1 = results1.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord1.index += 1 
results_ord1.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')



#using some hyperrameters to improve the performance of the model. the data used is the one that has been enhanced above not the raw data
tree_classifiers2 = {
  "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=0),
  "Extra Trees":ExtraTreesClassifier(max_depth=9, n_estimators=1000,random_state=0),
  "Random Forest":RandomForestClassifier(max_depth=9, n_estimators=1000,random_state=0),
  "AdaBoost":AdaBoostClassifier(learning_rate=0.1, n_estimators=1000, random_state=0),
  "LightGBM":LGBMClassifier(learning_rate=0.1, n_estimators=1000, random_state=0),
  "SVM":      SVC(kernel='linear', C=1000,random_state=0)}
tree_classifiers2 = {name: pipeline.make_pipeline( model) for name, model in tree_classifiers2.items()}

results2 = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in tree_classifiers2.items():
    
    start_time = time.time()
    model.fit(X_train1, y_train1)
    total_time = time.time() - start_time
        
    pred2 = model.predict(x_test1)
    
    results2 = results2.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test1, pred2)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test1, pred2)*100,
                              "Time":     total_time},
                              ignore_index=True)
results_ord2 = results2.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord2.index += 1 
results_ord2.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')




#importing joblib to train the best model

import joblib
best_model = tree_classifiers1.get("Random Forest")
best_model.fit(X_train1, y_train1)
joblib.dump(best_model, 'model.pkl')



benchmark= pd.read_csv('benmarch_model.csv')
feature= pd.read_csv('feature_engineer_model.csv')
hyper_param = pd.read_csv('hyper_param_model.csv')