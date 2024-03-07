# Car-reselling-price-predictor

import pandas as pd
import numpy as np
df=pd.read_csv('car data.csv')
df=pd.read_csv('car data.csv')
df.head()
df.tail()
df.shape
df.describe()
df.info()
df.isnull().sum()
df.head(1)

import datetime

df.head()

df.drop('Year',axis=1,inplace=True)
import seaborn as sns
df.shape

df.head(1)

df['Fuel_Type'].unique()
array(['Petrol', 'Diesel', 'CNG'], dtype=object)
'Petrol':1,'Diesel':2,'CNG':3
df['Fuel_Type']=df['Fuel_Type'].map({'Petrol':1,'Diesel':2,'CNG':3}).astype(int)
df['Seller_Type'].unique()
array(['Dealer', 'Individual'], dtype=object)
'Dealer':1,'Individual':2
df['Seller_Type']=df['Seller_Type'].map({'Dealer':1,'Individual':2}).astype(int)
df['Transmission'].unique()
array(['Manual', 'Automatic'], dtype=object)
'Manual':1,'Automatic':2
df['Transmission']=df['Transmission'].map({'Manual':1,'Automatic':2}).astype(int)
df['Owner'].unique()
array([0, 1, 3], dtype=int64)
df.head()

df.drop('Car_Name',axis=1,inplace=True)
df.head()

X = df.drop(['Selling_Price'],axis=1)
y = df['Selling_Price']
print(X)
print(y)
     Present_Price  Kms_Driven  Fuel_Type  Seller_Type  Transmission  Owner  \
0             5.59       27000          1            1             1      0   
1             9.54       43000          2            1             1      0   
2             9.85        6900          1            1             1      0   
3             4.15        5200          1            1             1      0   
4             6.87       42450          2            1             1      0   
..             ...         ...        ...          ...           ...    ...   
296          11.60       33988          2            1             1      0   
297           5.90       60000          1            1             1      0   
298          11.00       87934          1            1             1      0   
299          12.50        9000          2            1             1      0   
300           5.90        5464          1            1             1      0   

     car_Age  
0         10  
1         11  
2          7  
3         13  
4         10  
..       ...  
296        8  
297        9  
298       15  
299        7  
300        8  

#from sklearn.model_selection import train_test_split
#X_test,y_test,X_train,y_train=train_test_split(X,y,test_size=0.20,random_state=42)
#X_train = X_train.values.reshape(-1, 1) if len(X_train.shape) == 1 else X_train.values
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
lr = LinearRegression()
lr.fit(X, y)
​
rf = RandomForestRegressor()
rf.fit(X, y)
​
xgb = GradientBoostingRegressor()
xgb.fit(X, y)
​
xg = XGBRegressor()
xg.fit(X, y)
​
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, ...)
y_pred_lr=lr.predict(X)
y_pred_rf=rf.predict(X)
y_pred_xbg=xgb.predict(X)
y_pred_xg=xg.predict(X)
from sklearn import metrics
score1 = metrics.r2_score(y,y_pred_lr)
score2 = metrics.r2_score(y,y_pred_rf)
score3 = metrics.r2_score(y,y_pred_xbg)
score4 = metrics.r2_score(y,y_pred_xg)
print(score1,score2,score3,score4)
0.878075829051472 0.9944671642620836 0.9948207860112457 0.9999878414621973
final_data = pd.DataFrame({'Models':['LR','RF','GBR','XG'],
             "R2_SCORE":[score1,score2,score3,score4]})
final_data
Models	R2_SCORE
0	LR	0.878076
1	RF	0.994467
2	GBR	0.994821
3	XG	0.999988
sns.barplot(final_data['Models'],final_data['R2_SCORE'])

xg = XGBRegressor()
xg_final= xg.fit(X,y)
import joblib
joblib.dump(xg_final,'car_price_predictor')
model = joblib.load('car_price_predictor')
xg_final.save_model('xgb_model.json')
#joblib.dump(xg_final, 'xgb_model.pkl')
from tkinter import *
​
def show_entry_fields():
    try:
        p1=float(e1.get())
        p2=float(e2.get())
        p3=float(e3.get())
        p4=float(e4.get())
        p5=float(e5.get())
        p6=float(e6.get())
        p7=float(e7.get())
​
        # Load the model
        model = joblib.load('car_price_predictor')
​
        # Create a DataFrame with the input data
        data_new = pd.DataFrame({
            'Present_Price': [p1],
            'Kms_Driven': [p2],
            'Fuel_Type': [p3],
            'Seller_Type': [p4],
            'Transmission': [p5],
            'Owner': [p6],
            'Age': [p7]
        })
​
        # Predict the result
        result = model.predict(data_new)
​
        # Display the result
        Label(master, text="Car Purchase amount").grid(row=8)
        Label(master, text=result[0]).grid(row=10)
        print("Car Purchase amount", result[0])
    except Exception as e:
        print("An error occurred:", e)
​
master = Tk()
master.title("Car Price Prediction Using Machine Learning")
label = Label(master, text="Car Price Prediction Using Machine Learning", bg="black", fg="white")
label.grid(row=0, columnspan=2)
​
Label(master, text="Present_Price").grid(row=1)
Label(master, text="Kms_Driven").grid(row=2)
Label(master, text="Fuel_Type").grid(row=3)
Label(master, text="Seller_Type").grid(row=4)
Label(master, text="Transmission").grid(row=5)
Label(master, text="Owner").grid(row=6)
Label(master, text="Age").grid(row=7)
​
e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
​
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
Button(master, text='Predict', command=show_entry_fields).grid()
mainloop()
​
# Fuel_Type: 'Petrol':1,'Diesel':2,'CNG':3
# Seller_Type:'Dealer':1,'Individual':2
# Transmission:'Manual':1,'Automatic':2 
​
