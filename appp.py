import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import time

df = pd.read_csv('train_data.csv')
data=df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'user'], axis=1)
y = data['target']
x = data.drop(['target'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
tree_classifiers = {  "SVC": SVC(),
                    "Extra Trees":   ExtraTreesClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "AdaBoost":      AdaBoostClassifier(),
                    "Skl GBM":       GradientBoostingClassifier(),
                    "Logistic_reg":  LogisticRegression(),
                    "LightGBM":      KNeighborsClassifier(),
                    
                   
                    }

results = pd.DataFrame({'Model': [], 'AC':[], 'MSE': [], 'MAB': [], 'Time': []})

for model_name, model in tree_classifiers.items():
    start_time = time.time()
    model.fit(x_train,y_train)
    total_time = time.time()-start_time
    pred = model.predict(x_test)
    results = results.append({"Model":    model_name,
                            "AC": metrics.accuracy_score(y_test, pred)*100,
                            "MSE": metrics.mean_squared_error(y_test, pred),
                            "MAB": metrics.mean_absolute_error(y_test, pred),
                            "Time":     total_time},
                            ignore_index=True)

results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAB'], vmin=0, vmax=100, color='#5fba7d')
                    