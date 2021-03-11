import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

def MAE_cal(max_leaf_nodes,train_X,valid_X,train_Y,valid_Y):
    MOX_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    MOX_model.fit(train_X, train_Y.astype('int'))
    val_prediction = MOX_model.predict(valid_X)
    res = mean_absolute_error(valid_Y, val_prediction)
    return res

#T for training, V for validating
MOX_path="./MOX Conclusion.csv"
MOX_data=pd.read_csv(MOX_path)
MOX_features=["Humidity"
              ,"R1_Up_Slope","R1_Down_Slope","R2_Up_Slope","R2_Down_Slope","R3_Up_Slope","R3_Down_Slope"
              ,"R4_Up_Slope","R4_Down_Slope","R5_Up_Slope","R5_Down_Slope","R6_Up_Slope","R6_Down_Slope"
              ,"R7_Up_Slope","R7_Down_Slope","R8_Up_Slope","R8_Down_Slope","R9_Up_Slope","R9_Down_Slope"
              ,"R10_Up_Slope","R10_Down_Slope","R11_Up_Slope","R11_Down_Slope","R12_Up_Slope","R12_Down_Slope"
              ,"R13_Up_Slope","R13_Down_Slope","R14_Up_Slope","R14_Down_Slope"]
X=MOX_data[MOX_features] # X as feature
Y=MOX_data.CO_Concentration # Y as prediction target

#MAE
split_num=10
K_split=KFold(n_splits=split_num)
MAE_table=[]
for i in [10,50,100,500,2000,5000]:
    sum_MAE = 0
    count = 0
    print("\nMax leaf nodes:", i)
    for train_part, valid_part in K_split.split(X, Y):
        count += 1
        train_X = X.iloc[train_part]
        valid_X = X.iloc[valid_part]
        train_Y = Y.iloc[train_part]
        valid_Y = Y.iloc[valid_part]
        current_MAE = MAE_cal(i,train_X, valid_X, train_Y, valid_Y)
        MAE_table.append(current_MAE)
        sum_MAE += current_MAE
        print(count, " Current MAE= ", current_MAE)
    MAE_mean = round(sum_MAE / split_num, 2)
    MAE_std = round(np.std(MAE_table), 2)
    print("MAE mean:", MAE_mean)
    print("MAE standard deviation:", MAE_std)

#Classification Accuracy
for i in [10,50,100,500,2000,5000]:
    print("\nMax leaf nodes:", i)
    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(DecisionTreeClassifier(max_leaf_nodes=i), X, Y.astype('int'), cv=kfold, scoring='accuracy')
    print(results)
    print(("Classification Accuracy: mean=%.3f, standard deviation=%.3f") % (results.mean(), results.std()))