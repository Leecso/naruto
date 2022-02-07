import pandas as pd
import sys
dir_route = sys.argv[1]
datas = pd.read_csv(dir_route+'/train.txt',sep=',', names=[
                                                           'ctm_cid', 'label', 'order_30', 'order_15', 'order_7', 'order_3',
                                                           'order_1', 'chart_30', 'chart_15', 'chart_7', 'chart_3', 'chart_1',
                                                           'click_30', 'click_15', 'click_7', 'click_3', 'click_1'
                                                          ],header=None)
datas=datas.fillna(0)
from sklearn.ensemble import GradientBoostingRegressor

print (datas.shape)
print (datas.columns)

train_X = datas[['order_30', 'order_15', 'order_7', 'order_3',
       'order_1', 'chart_30', 'chart_15', 'chart_7', 'chart_3', 'chart_1',
       'click_30', 'click_15', 'click_7', 'click_3', 'click_1']]
train_Y = datas['label']
regressor = GradientBoostingRegressor(n_estimators=100,
                                 learning_rate=1.0,
                                 max_depth=5,
                                 random_state=0).fit(train_X, train_Y)

def sigmoid(x):
    return 1/(1+np.exp(-x))

import numpy as np
predict_datas = pd.read_csv(dir_route+"/predict.txt",sep=",", names=[
                                                           'ctm_cid', 'order_30', 'order_15', 'order_7', 'order_3',
                                                           'order_1', 'chart_30', 'chart_15', 'chart_7', 'chart_3', 'chart_1',
                                                           'click_30', 'click_15', 'click_7', 'click_3', 'click_1'
                                                          ],header=None)
predict_datas = predict_datas.fillna(0)
print(predict_datas.shape)

predict_X = predict_datas[['order_30', 'order_15', 'order_7', 'order_3',
       'order_1', 'chart_30', 'chart_15', 'chart_7', 'chart_3', 'chart_1',
       'click_30', 'click_15', 'click_7', 'click_3', 'click_1']]
result=regressor.predict(predict_X)
print("result size:"+str(result.shape))
result_normalization=sigmoid(np.log((result-min(result))/(max(result)-min(result))*50+1.1))
keys = predict_datas['ctm_cid']

result_file = open(dir_route+"/predict_result.csv","w+")
for i in range(len(keys)):
    result_file.writelines(str(keys[i])+","+str(result_normalization[i])+"\n")
result_file.close()

