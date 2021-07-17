import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

data0 = pd.read_csv('./202011_data.csv')

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

data = data0.drop(['Reach12+', 'Reach15+', 'year', 'new_reg', 'after_1m',
       'after_2m', 'after_3m', 'after_4m', 'after_5m', 'after_6m','log_0', 'log_1', 'log_2', 'log_3', 'log_4', 'log_5',
       'log_6',], 1)

data2 = ss_f.transform(data)

# modeldirs = ["./elu_SGD_final/0_best_model.h5",
#              "./elu_adam_final/1_best_model.h5",
#              "./elu_SGD_final/2_best_model.h5",
#              "./elu_adam_final/3_best_model.h5",
#              "./elu_adam_final/4_best_model.h5",
#              "./elu_adam_final/5_best_model.h5",
#              "./elu_SGD_final/6_best_model.h5",


model = load_model("./elu_SGD_final/5_best_model.h5")
data0['pred_5m'] =  np.exp(model.predict(data2))


print(data0[['pred_5m']].iloc[0:10])
print('5개월후 2021_04월 Camry의 개인 예측 등록대수는 : ', data0['pred_5m'].iloc[0:10].sum(), '대 입니다')
#data0.to_csv('./202011_data_pred.csv', index=False)


