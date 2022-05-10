import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno
from sklearn.preprocessing import OrdinalEncoder
five_second_data = pd.read_csv("./dataset_5secondWindow/dataset_5secondWindow.csv")

not_null_col = [col for col in five_second_data.columns if five_second_data[col].isnull().sum()<800]
new_data = pd.DataFrame(five_second_data[not_null_col])
print(new_data.head())

OE = OrdinalEncoder()
ct = np.asarray(new_data['target'])
new_data['target'] = OE.fit_transform(ct.reshape(1,-1))
print(new_data)

