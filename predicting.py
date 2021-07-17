# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import math

data = pd.read_csv('./2021_4.csv')


data = data.dropna() #파일 상단의 파일 정보 때문에 생긴 nan값 없애기 
data.columns = list(data.iloc[0]) #컬럼 재설정하기 
data = data.drop(index={27}) #불필요한 row 없애기 

#data.reset_index(drop=True, inplace=True)
#전처리 : 한자릿수인 월은 0을 더해서 두자릿수로 표시(4월 -> 04월), 나이에서 60대 이상은 60대로 
data['타겟\변수'] = data['타겟\변수'].apply(lambda x : x.replace('이상',''))
data['월'] = data['월'].apply(lambda x : x.zfill(3))
#컬럼명 맞추기 
data = data.rename(columns={'월' : 'Month', '타겟\변수' : 'ad_id', '광고횟수' : 'sp', 'GRP' : 'grp', '총시청자수' : 'Aud', '도달율 1+' : 'Reach1+', '도달율 3+' : 'Reach3+', '도달율 5+' : 'Reach5+', '도달율 7+' : 'Reach7+', '도달율 9+' : 'Reach9+', '도달율 12+' : 'Reach12+'})


#데이터타입 -> float 
data['sp'] = data['sp'].astype(float)
data['grp'] = data['grp'].astype(float)
data['Aud'] = data['Aud'].astype(float)
data['Reach1+'] = data['Reach1+'].astype(float)
data['Reach3+'] = data['Reach3+'].astype(float)
data['Reach5+'] = data['Reach5+'].astype(float)
data['Reach7+'] = data['Reach7+'].astype(float)
data['Reach9+'] = data['Reach9+'].astype(float)
data['Reach12+'] = data['Reach12+'].astype(float)

#data = data[data['Reach1+'] != 0]
#data.reset_index(drop=True, inplace=True)

#보관할 데이터프레임 남겨놓기 
keepdf = data[['년', "Month", 'Advertiser', 'Product', 'ad_id']].copy()



data.drop(['년', 'Advertiser', 'Product'],1, inplace=True)
data = pd.get_dummies(data, columns={'Month', 'ad_id'})


#4월 데이터만 있으므로 get_dummies로 다른 월의 정보가 만들어지지 않아서 컬럼 만들어주는 과정 
data['Month_01월'] = 0
data['Month_02월'] = 0
data['Month_03월'] = 0
#data['Month_04월'] = 0
data['Month_05월'] = 0
data['Month_06월'] = 0
data['Month_07월'] = 0
data['Month_08월'] = 0
data['Month_09월'] = 0
data['Month_10월'] = 0
data['Month_11월'] = 0
data['Month_12월'] = 0

data['ZSCORE'] = (data['grp'] - data.describe()['grp']['mean'])/data.describe()['grp']['std']

# Reach12+ 삭제할지 말지
data.drop(['Reach12+'],1,inplace=True)

# 브랜드 데이터 불러오기 - scaler를 사용하기 위함
car_df = pd.read_csv('./toyota_new.csv') 


mm = MinMaxScaler()
ss = StandardScaler()
rs = RobustScaler()


# 브랜드 데이터프레임 전처리
# car_df['ad_id'] = car_df['ad_id'].apply(lambda x : x.replace('이상',''))
# car_df['ad_id'] = car_df['ad_id'].apply(lambda x : x[-4:])
# car_df = pd.get_dummies(car_df, columns={'ad_id', 'Month'})

# car_df['log_0'] = np.log(car_df['new_reg']+1)
# car_df['log_1'] = np.log(car_df['after_1m']+1) 
# car_df['log_2'] = np.log(car_df['after_2m']+1) 
# car_df['log_3'] = np.log(car_df['after_3m']+1) 
# car_df['log_4'] = np.log(car_df['after_4m']+1) 
# car_df['log_5'] = np.log(car_df['after_5m']+1) 
# car_df['log_6'] = np.log(car_df['after_6m']+1)

# car_df['ZSCORE'] = (car_df['grp'] - car_df.describe()['grp']['mean'])/car_df.describe()['grp']['std']


X = car_df.drop(["year", 'new_reg', 'after_1m', 
       'after_2m', 'after_3m', 'after_4m', 'after_5m', 'after_6m','log_0', 'log_1', 'log_2', 'log_3', 'log_4',
       'log_5', 'log_6', "Reach15+", "Reach12+"],1)
y = car_df['log_0']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, shuffle=True)

ss_f = ss.fit(X_train)




pred = ss_f.transform(data)

from tensorflow.keras.models import load_model

## 파일명 및 경로수정 필요 !!! 
modeldirs = ["./elu_SGD_final/0_best_model.h5",
             "./elu_adam_final/1_best_model.h5",
             "./elu_SGD_final/2_best_model.h5",
             "./elu_adam_final/3_best_model.h5",
             "./elu_adam_final/4_best_model.h5",
             "./elu_adam_final/5_best_model.h5",
             "./elu_SGD_final/6_best_model.h5",
             ]
for v, i in enumerate(modeldirs): 
  model = load_model(f"{i}")

  register = []
  for a in range(0,len(pred)):
    register.append(math.exp(model.predict(pred[[a]])))

  # data['predict'] = register
  # print(np.array(register).mean())

  data[f'predict_reg_{v}'] = register


prediction = pd.concat([keepdf, data], axis=1)
prediction.to_csv("./prediction_final.csv", index=False, encoding='cp949')

