# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
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
y = toyota_df['log_0']
i=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13, shuffle=True)

ss_f = ss.fit(X_train)
X_train = ss_f.transform(X_train)
X_test = ss_f.transform(X_test)

model = keras.Sequential([

layers.Dense(16, activation='elu', input_shape=list(X_train.shape)),  
    
layers.Dense(8, activation='elu'),
    
layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=300, verbose=1, mode='min', baseline=None, restore_best_weights=False)
mc = ModelCheckpoint(f'{i}_best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

history = model.fit(
X_train, y_train, batch_size= 5,
epochs= 10000 , verbose=1, validation_data=(X_test, y_test),callbacks=[early_stopping, mc])


my_Work = pd.DataFrame(history.history)

my_Work["loss"].plot(color="r")
my_Work["val_loss"].plot(color="g")
plt.legend()
plt.title('val_loss : ' + str(my_Work['val_loss'][-1:].values[0]), pad=30)
#loss, mae, mse
plt.savefig(f"./{i}.png")
plt.show()