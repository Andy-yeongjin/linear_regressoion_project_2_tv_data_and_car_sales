from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.keras.backend import to_dense

test = pd.read_csv('./toyota_test.csv')

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

test0 = test.drop(['new_reg', 'after_1m', 'after_2m', 'after_3m', 'after_4m',
       'after_5m', 'after_6m', 'log_0',
       'log_1', 'log_2', 'log_3', 'log_4', 'log_5', 'log_6'], 1)

test2 = ss_f.transform(test0)
test_ori = test[['new_reg', 'after_1m', 'after_2m', 'after_3m', 'after_4m', 'after_5m']]+1
test_ori.to_csv('./test_ori.csv', index=False)

model = load_model("./elu_SGD_final/0_best_model.h5")
reg_0 = test[['new_reg']]+1 - np.exp(model.predict(test2))
test_ori['pred'] = np.exp(model.predict(test2))

model = load_model("./elu_adam_final/1_best_model.h5")
reg_1 = test[['after_1m']]+1 - np.exp(model.predict(test2))
test_ori['pred_1m'] = np.exp(model.predict(test2))

model = load_model("./elu_SGD_final/2_best_model.h5")
reg_2 = test[['after_2m']]+1 - np.exp(model.predict(test2))
test_ori['pred_2m'] = np.exp(model.predict(test2))

model = load_model("./elu_adam_final/3_best_model.h5")
reg_3 = test[['after_3m']]+1 - np.exp(model.predict(test2))
test_ori['pred_3m'] = np.exp(model.predict(test2))

model = load_model("./elu_SGD_final/4_best_model.h5")
reg_4 = test[['after_4m']]+1 - np.exp(model.predict(test2))
test_ori['pred_4m'] = np.exp(model.predict(test2))

model = load_model("./elu_adam_final/5_best_model.h5")
reg_5 = test[['after_5m']]+1 - np.exp(model.predict(test2))
test_ori['pred_5m'] = np.exp(model.predict(test2))

# model = load_model("./elu_SGD_final/6_best_model.h5")
# reg_6 = test[['after_6m']]+1 - np.exp(model.predict(test2))

# pred = pd.concat([reg_0,reg_1,reg_2,reg_3,reg_4,reg_5],1)

#pred.to_csv('./pred.csv', index=False)
test_ori.to_csv('./test_ori_pred.csv', index=False)

# print(pred)