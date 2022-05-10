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
five_second_data = pd.read_csv("./dataset_5secondWindow/dataset_5secondWindow.csv")

not_null_col = [col for col in five_second_data.columns if five_second_data[col].isnull().sum()<800]
new_data = pd.DataFrame(five_second_data[not_null_col])
print(new_data.head())

OE = OrdinalEncoder()
ct = np.asarray(new_data['target'])
new_data['target'] = OE.fit_transform(ct.reshape(-1,1))


ct2 = np.asarray(new_data['user'])
new_data['user'] = OE.fit_transform(ct2.reshape(-1,1))
new_data = new_data.sort_values(by='user', ascending=True)
print(new_data)

# df = pd.read_csv("./final_dataset.csv")
# #print(df.shape)
# #split original data into train and test
# df_train = df.iloc[:4714,:]
# df_test = df.iloc[4714:,:]
# #print(df_test.shape)

# #Train set 
# X_train = df_train.drop(['target'],axis=1)
# y_train = df_train['target']

# # Test set 

# x_test = df_test.drop(['target'],axis =1)
# y_test = df_test['target']

# #scaled the features
# SC = StandardScaler()
# X_train = SC.fit_transform(X_train)
# x_test = SC.transform(x_test)
# #print(X_train)

# tree_classifiers = {
#   "Decision Tree": DecisionTreeClassifier(),
#   "Extra Trees":ExtraTreesClassifier(),
#   "Random Forest":RandomForestClassifier(),
#   "AdaBoost":AdaBoostClassifier(),
#   "Skl GBM": GradientBoostingClassifier(),
#   "Skl HistGBM":HistGradientBoostingClassifier(),
#   "XGBoost": XGBClassifier(),
#   "LightGBM":LGBMClassifier(),
#   "CatBoost": CatBoostClassifier(),
#   "SVM":      SVC()}
# tree_classifiers = {name: pipeline.make_pipeline( model) for name, model in tree_classifiers.items()}

# results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
# for model_name, model in tree_classifiers.items():
    
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     total_time = time.time() - start_time
        
#     pred = model.predict(x_test)
    
#     results = results.append({"Model":    model_name,
#                               "Accuracy": metrics.accuracy_score(y_test, pred)*100,
#                               "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
#                               "Time":     total_time},
#                               ignore_index=True)
                              
                              
# results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
# # results_ord.index += 1 
# # results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
# print(results_ord)
