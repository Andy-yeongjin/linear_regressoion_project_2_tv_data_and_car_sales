import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 기존 파일, 202011월달이 있는 파일
origin = pd.read_csv('./toyota_a.csv')

data = origin[(origin['year'] == 2020) & (origin['Month_11월'] == 1)] 

data = data[data['grp'] != 0]
data.reset_index(drop=True, inplace=True)

data2 = data.drop(['Reach12+', 'Reach15+', 'year', 'new_reg', 'after_1m',
       'after_2m', 'after_3m', 'after_4m', 'after_5m', 'after_6m', 'log_0', 'log_1', 'log_2', 'log_3', 'log_4', 'log_5',
       'log_6'], 1)


# 새로운 파일, 202011부터 202105까지 지운 파일
# 브랜드 데이터 불러오기 - scaler를 사용하기 위함
car_df = pd.read_csv('./toyota_new.csv') 


mm = MinMaxScaler()
ss = StandardScaler()
rs = RobustScaler()


X = car_df.drop(["year", 'new_reg', 'after_1m', 
       'after_2m', 'after_3m', 'after_4m', 'after_5m', 'after_6m','log_0', 'log_1', 'log_2', 'log_3', 'log_4',
       'log_5', 'log_6', "Reach15+", "Reach12+"],1)
y = car_df['log_0']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, shuffle=True)

ss_f = ss.fit(X_train)

from tensorflow.keras.models import load_model

data2 = ss_f.transform(data2)

# modeldirs = ["./elu_SGD_final/0_best_model.h5",
#              "./elu_adam_final/1_best_model.h5",
#              "./elu_SGD_final/2_best_model.h5",
#              "./elu_adam_final/3_best_model.h5",
#              "./elu_adam_final/4_best_model.h5",
#              "./elu_adam_final/5_best_model.h5",
#              "./elu_SGD_final/6_best_model.h5",

model = load_model("./elu_SGD_final/0_best_model.h5")
reg_0 = data[['new_reg']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_adam_final/1_best_model.h5")
reg_1 = data[['after_1m']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_SGD_final/2_best_model.h5")
reg_2 = data[['after_2m']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_adam_final/3_best_model.h5")
reg_3 = data[['after_3m']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_adam_final/4_best_model.h5")
reg_4 = data[['after_4m']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_adam_final/5_best_model.h5")
reg_5 = data[['after_5m']]+1 - np.exp(model.predict(data2))

model = load_model("./elu_SGD_final/6_best_model.h5")
reg_6 = data[['after_6m']]+1 - np.exp(model.predict(data2))

pred = pd.concat([reg_0,reg_1,reg_2,reg_3,reg_4,reg_5],1)

data.to_csv('./202011_data.csv')
pred.to_csv('./202011_predict.csv')

