# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



mm = MinMaxScaler()
ss = StandardScaler()
rs = RobustScaler()


toyota_df = pd.read_csv('./toyota_new.csv')

X = toyota_df.drop(['Reach12+','Reach15+', 'new_reg', 'after_1m', 'year',
       'after_2m', 'after_3m', 'after_4m', 'after_5m', 'after_6m','log_0', 'log_1', 'log_2', 'log_3', 'log_4',
       'log_5', 'log_6'],1)
y = toyota_df['log_6']
i = 6

from tensorflow.keras.models import load_model
model = load_model(f'./elu_SGD_final/{i}_best_model.h5')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13, shuffle=True)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=23, shuffle=True)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=63, shuffle=True)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=0.3, random_state=43, shuffle=True)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size=0.3, random_state=53, shuffle=True)

ss_f = ss.fit(X_train)
X_train = ss_f.transform(X_train)
X_test = ss_f.transform(X_test)

# 교차검증
X_test2 = ss_f.transform(X_test2)
X_test3 = ss_f.transform(X_test3)
X_test4 = ss_f.transform(X_test4)
X_test5 = ss_f.transform(X_test5)


result = str(model.evaluate(X_test, y_test)[0])
result2 = str(model.evaluate(X_test2, y_test2)[0])
result3 = str(model.evaluate(X_test3, y_test3)[0])
result4 = str(model.evaluate(X_test4, y_test4)[0])
result5 = str(model.evaluate(X_test5, y_test5)[0])


print('test_mse, test_mae: ', result)
print('test_mse2, test_mae2: ', result2)
print('test_mse3, test_mae3: ', result3)
print('test_mse4, test_mae4: ', result4)
print('test_mse5, test_mae5: ', result5)


f = open(f'{i}_model_kfold.txt', mode='wt', encoding='utf-8')
f.write('X_test : ')
f.write(result)
f.write('\n')
f.write('X_test2 : ')
f.write(result2)
f.write('\n')
f.write('X_test3 : ')
f.write(result3)
f.write('\n')
f.write('X_test4 : ')
f.write(result4)
f.write('\n')
f.write('X_test5 : ')
f.write(result5)
f.write('\n')

f.close()