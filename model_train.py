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
# five_second_data = pd.read_csv("./dataset_5secondWindow/dataset_5secondWindow.csv")

# not_null_col = [col for col in five_second_data.columns if five_second_data[col].isnull().sum()<800]
# new_data = pd.DataFrame(five_second_data[not_null_col])
# print(new_data.head())
OE = OrdinalEncoder()
# ct = np.asarray(new_data['target'])
# new_data['target'] = OE.fit_transform(ct.reshape(-1,1))


# print(new_data)
df = pd.read_csv("./fine1_dataset.csv")

ct2 = np.asarray(df['user'])
df['user'] = OE.fit_transform(ct2.reshape(-1,1))
df = df.sort_values(by='user', ascending=True)
#print(df)

# df = pd.read_csv("./final_dataset.csv")
# #print(df.shape)
#split original data into train and test
df_train = df.iloc[:4680,:]
df_test = df.iloc[4680:,:]
#print(df_train)

#Train set 
X_train = df_train.drop(['target','user'],axis=1)
#removing id column and unnamed column. not a feature needed for the training
X_train = X_train.iloc[:,2:]

y_train = df_train['target']
#print(X_train)
# Test set 

x_test = df_test.drop(['target','user'],axis =1)
x_test= x_test.iloc[:,2:]
y_test = df_test['target']

#scaled the features
# SC = StandardScaler()
# X_train = SC.fit_transform(X_train)
# x_test = SC.transform(x_test)
#print(X_train)

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
# results_ord.index += 1 
# results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
print(results_ord)

#the accuracy result is quite low, run gridsearch on some of the model according to the accuracy and time. we are more concern about how fast the model is because it will be run on a watch


# param_DecisionTree  = {'max_depth'   : [3, 4, 5, 6, 7, 8, 9]}
# param_ExtraTree     = {'n_estimators': [1, 10, 100, 100, 1000],     'max_depth'     : [3, 4, 5, 6, 7, 8, 9]}
# param_RandomForest  = {'n_estimators': [1, 10, 100, 100, 1000],     'max_depth'     : [3, 4, 5, 6, 7, 8, 9]}
# param_AdaBoost      = {'n_estimators': [1, 10, 100, 100, 1000],     'learning_rate' : [0.01, 0.1, 1, 10, 100, 1000]}
# param_LightGBM      = {'n_estimators': [1, 10, 100, 1000],    'learning_rate' : [0.01, 0.1, 1, 10, 100, 1000]}
# param_SVC           = {'kernel'      : ['linear', 'rbf'],                  'C'           : [0.01, 0.1, 1, 10, 100, 1000]}      # 'gamma'         : [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,      


# grid_search_DecisionTree      = GridSearchCV(DecisionTreeClassifier(), param_DecisionTree, cv=5)
# grid_search_ExtraTree         = GridSearchCV(ExtraTreesClassifier(), param_ExtraTree, cv=5)
# grid_search_RandomForest      = GridSearchCV(RandomForestClassifier(), param_RandomForest, cv=5)
# grid_search_AdaBoost          = GridSearchCV(AdaBoostClassifier(), param_AdaBoost, cv=5)
# grid_search_LightGBM          = GridSearchCV(LGBMClassifier(), param_LightGBM, cv=5)
# grid_search_SVC               = GridSearchCV(SVC(), param_SVC, cv=5)

# grid_search_DecisionTree.fit(X_train, y_train)
# grid_search_ExtraTree.fit(X_train, y_train)
# grid_search_RandomForest.fit(X_train, y_train)
# grid_search_AdaBoost.fit(X_train, y_train)
# grid_search_LightGBM.fit(X_train, y_train)
# grid_search_SVC.fit(X_train, y_train)

# print(f'Decision Tree    =>    Best Score : {round(grid_search_DecisionTree.best_score_ * 100, 2)}    Best Parameters :   {grid_search_DecisionTree.best_params_}  ')
# print(f'Extra Trees      =>    Best Score : {round(grid_search_ExtraTree.best_score_ * 100, 2)}    Best Parameters :   {grid_search_ExtraTree.best_params_}  ')
# print(f'Random Forest    =>    Best Score : {round(grid_search_RandomForest.best_score_ * 100, 2)}    Best Parameters :   {grid_search_RandomForest.best_params_}  ')
# print(f'AdaBoost         =>    Best Score : {round(grid_search_AdaBoost.best_score_ * 100, 2)}    Best Parameters :   {grid_search_AdaBoost.best_params_}  ')
# print(f'LightGBM         =>    Best Score : {round(grid_search_LightGBM.best_score_ * 100, 2)}    Best Parameters :   {grid_search_LightGBM.best_params_}  ')
# print(f'SVC              =>    Best Score : {round(grid_search_SVC.best_score_ * 100, 2)}    Best Parameters :   {grid_search_SVC.best_params_}  ')

#imputing the hyperparameters to the selected model, after grid_search on the bestparam
tree_classifiers2 = {
  "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=0),
  "Extra Trees":ExtraTreesClassifier(max_depth=9, n_estimators=1000,random_state=0),
  "Random Forest":RandomForestClassifier(max_depth=9, n_estimators=1000,random_state=0),
  "AdaBoost":AdaBoostClassifier(learning_rate=0.1, n_estimators=1000, random_state=0),
  "LightGBM":LGBMClassifier(learning_rate=0.1, n_estimators=1000, random_state=0),
  "SVM":      SVC(kernel='linear', C=1000,random_state=0)}
tree_classifiers2 = {name: pipeline.make_pipeline( model) for name, model in tree_classifiers2.items()}

results1 = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in tree_classifiers2.items():
    
    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_test)
    
    results1 = results1.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
results_ord1 = results1.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
# # results_ord.index += 1 
# # results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
print(results_ord1)
                              