import numpy as np
import pandas as pd

data = pd.read_csv('datan.csv')
# print (data)
data = data.drop(columns=['area_type','availability','society','balcony'])
# print(data)

data['location']=data['location'].fillna('unknown')
data['size'].fillna(data['size'].mode()[0],inplace=True)
data['bath'].fillna(data['bath'].median(),inplace=True)

print(data.isnull().sum())
# print(data)

def convert_sqrt(x):
    try:
        if '-' in str(x):
            a,b = x.split('-')
            return (float(a)+float(b))/2
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqrt)
data = data.dropna(subset=['total_sqft'])
    

def extract_num(x):
    try:
        return int(str(x).split(" ")[0])
    except:
        return None
data['BHK_or_Bedroom'] = data['size'].apply(extract_num)
data = data.drop(columns=["size"])

data =data[(data['total_sqft']/data['BHK_or_Bedroom'])<1000]

data['price_per_sqft']= data['price']/data['total_sqft']

data = pd.get_dummies(data,columns=['location'],drop_first=True)
# data.to_csv('clean_data.csv',index=False)
# print('data clean hogya h: clean_data.csv ')

from sklearn.preprocessing import MinMaxScaler

X = data.drop(columns=['price'])
Y = data['price']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train,x_test,y_train,y_test = train_test_split(X_scaled,Y,test_size=0.2,random_state=42)

model = RandomForestRegressor(random_state=42,n_estimators=100)
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

mae = mean_absolute_error(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_predict)

print(mae)
print(mse)
print(rmse)
print(r2)
