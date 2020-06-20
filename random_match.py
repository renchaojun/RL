import random
import matplotlib.pyplot as plt
import math, random
import numpy as np
import inputdata
import Utils
ar=[[0,0,0,0]for i in range(10)]
for i in range(10):
    index=random.randint(2,4)
    ar[i][index-1]=1
arr=[]
arr.append(ar)
origin_data = inputdata.get_data()
train_data=inputdata.get_input_data(origin_data)
split=int(len(origin_data)*0.8)
xunlian_data=origin_data[0:split]
pre_data=origin_data[split:len(origin_data)]

pre_time=[]
xunlian_time=[]
#训练组80%的数据
for i in range(len(xunlian_data)):
    xunlian_adata = []
    xunlian_adata.append(np.array(xunlian_data[i]))
    pici = inputdata.cul_pici(arr, 1)
    dic, b_dic = inputdata.daikuan(xunlian_adata, pici)
    # print("每个批次任务的带宽分配情况：", b_dic)
    time_ = inputdata.jisuan(xunlian_adata, dic, b_dic)
    xunlian_time.append(time_)
print("随机匹配的训练80%任务的时间列表长度",len(xunlian_time))
Utils.save("random_xunlian80.npy", xunlian_time)

#预测后20%组数据
for i in range(len(pre_data)):
    pre_adata = []
    pre_adata.append(np.array(pre_data[i]))
    pici = inputdata.cul_pici(arr, 1)
    dic, b_dic = inputdata.daikuan(pre_adata, pici)
    # print("每个批次任务的带宽分配情况：", b_dic)
    time_ = inputdata.jisuan(pre_adata, dic, b_dic)
    pre_time.append(time_)
print("随机匹配的预测20%任务的时间列表长度",len(pre_time))
Utils.save("random_pre20.npy", pre_time)



yichuan_xunlian_time=Utils.load("xunliantime.npy")
yichuan_xunlian_time=yichuan_xunlian_time[0:split]
random_xunlian_time=Utils.load("random_xunlian80.npy")

##画图  遗传算法和随机匹配的标签
# xx = [i for i in range(len(yichuan_xunlian_time))]
xx = [i for i in range(len(yichuan_xunlian_time))]
print("标签的误差情况",(yichuan_xunlian_time-random_xunlian_time))
plt.plot(xx, yichuan_xunlian_time)
plt.plot(xx, random_xunlian_time)
plt.legend(['yichuan', 'random'], loc='upper right')
plt.show()