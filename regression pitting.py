# -- coding: utf-8 --
"""
Created on Wed Jan  5 10:41:56 2022

@author: sutha
"""

# -- coding: utf-8 --
"""
Created on Tue Jan  4 15:30:55 2022

@author: sutha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r'C:/Users/sutha/Downloads/Final - Sheet1.csv')
df.drop([0],axis=0,inplace=True)

#df.iloc[:,2] = pd.get_dummies(df.iloc[:,2],drop_first = True)
#temp_place = pd.get_dummies(df.iloc[:,1],drop_first = True)
#df.drop(columns = "Place",inplace = True)
#df = pd.concat([df,temp_place],axis=1)

df['Si'] = df['Si'].replace("<0.003",0.003)
df['Mn'] = df['Mn'].replace("<0.01",0.01)
df['Cu'] = df['Cu'].replace("<0.01",0.01)
df['Cr'] = df['Cr'].replace("<0.01",0.01)
df['Ni'] = df['Ni'].replace("<0.01",0.01)
df['Al'] = df['Al'].replace("<0.01",0.01)
df['Ti'] = df['Ti'].replace("<0.01",0.01)
df['V'] = df['V'].replace("<0.01",0.01)

for data in df["Corrosion"]:
    if type(data)==str and data[0]==">":
        df["Corrosion"]=df["Corrosion"].replace(data,float(data[1:]))

col = ['C', 'Si', 'Mn', 'P', 'S', 'Cu', 'Cr','Ni', 'Al', 'Ti', 'V','Corrosion']
for i in col:
    df[i] = pd.to_numeric(df[i])

c=list(df.columns)



p=[0, 2, 3, 5, 6, 7, 9, 16, 17, 18, 22]

X = df.iloc[:,df.columns!='Corrosion'].values
y = pd.DataFrame(df.iloc[:,-1].values)


le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])
X[:,1]=le.fit_transform(X[:,1])
X[:,2]=le.fit_transform(X[:,2])



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X)
  
X = imputer.transform(X)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
y.fillna(value=y.mean(), inplace=True)
y=y.values.ravel()
y_scaled=scaler.fit_transform(y.reshape(-1,1))
y_scaled=y_scaled.ravel()


regressor = [["RFR",RandomForestRegressor(n_estimators=100,random_state=0)],["SVR",SVR(kernel='rbf',gamma='auto',cache_size=500)],["MLR",LinearRegression(n_jobs=100,normalize=False)],["Hybrid Model"]]
ans = []
predictions={"RFR":[],"MLR":[],"SVR":[]}
for i in range(len(regressor)-1):
    temp=[]
    for j in range(50,100,10):
        X_train,X_test,y_train,y_test = train_test_split(X,y_scaled,test_size=(100-j)/100,random_state=0)
        
        regressor[i][1].fit(X_train,y_train)
        y_pred = regressor[i][1].predict(X_test)
        predictions[regressor[i][0]].append(y_pred)
        acc = r2_score(y_test,y_pred)
        
        #acc = cross_val_score(regressor[i][1],X_train,y_train,cv=10)
        temp.append(np.mean(acc))
    ans.append(temp)

temp=[]
for j in range(50,100,10):
    X_train,X_test,y_train,y_test = train_test_split(X,y_scaled,test_size=(100-j)/100,random_state=0)
    y_pred= (0.92*predictions["RFR"][j//10-5]+0.05*predictions["MLR"][j//10-5]+0.36*predictions["SVR"][j//10-5])
    acc=r2_score(y_test,y_pred)
    
    temp.append(np.mean(acc))
    
  
ans.append(temp)
    
    
    
    
    


colour = ['blue','green','red','black']

train = np.arange(50,100,10)

importance=regressor[0][1].feature_importances_
plt.bar([x for x in range(len(importance))], importance)
plt.hlines(0.01,- 0, 24, color='red',linestyle='--')
plt.show()

for i in range(4):
    plt.plot(train,ans[i],color=colour[i],label=regressor[i][0])

plt.title("Pitting with cv")
plt.xlabel("Train %")
plt.ylabel("r2 score")
plt.legend()
plt.show()