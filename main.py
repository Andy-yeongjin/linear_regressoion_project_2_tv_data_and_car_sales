import pandas as pd
import matplotlib.pyplot as plt


import platform
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

# 한글 사용시 마이너스 폰트가 깨지는 문제가 발생할 수 있으므로 설정변경
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Linux':
    rc('font', family='NanumBarunGothic')
else:
    print('Unknown system... sorry~~')


history = pd.read_csv("history.csv")
x = []
y = []
k = [] 


for i in range(1800):

    x.append(history.index[i])
    y.append(history["loss"][i])
    k.append(history["val_loss"][i])
    
    if i % 5 ==0:
        plt.ylim(0,10)
        plt.xlim(0,1800)
        plt.xlabel("반복 시행 횟수")
        plt.ylabel("오차율 ")
        plt.plot(x, y , color="red", label="loss")
        plt.plot(x, k , color="blue", label="val_loss")
        plt.pause(0.001)
plt.show()
