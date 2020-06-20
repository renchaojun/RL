import numpy as np
import  random
from random import choice
import Utils

num_size = 60000  # 生成多少数据
if __name__ == '__main__':

    #wf=。*L, L就是数据大小, f就是10^10,

    ##1.随机生成数据表示数据的大小L
    L= np.around(np.random.normal(loc=1000,scale=100,size=num_size),decimals=0) #Kb单位，均值mean,标准差std,数量
    print("正太分布:",L)
    ##gamma分布产生关系值
    F = np.around([random.gammavariate(4, 200) for _ in range(0, num_size)],decimals=0)
    print("gamma分布",F)
    #需要的计算周期 单位*10的10 cycles
    w=10**6*L*F/(10**10)     ##(ms)  10**6是系数
    print(w) ## 4ms

    #2.带宽
    daikuan=[40,60,100]  ##数据大小（）/（最大传输速率 8 bit/s*带宽hz）
    D=[choice(daikuan) for i in range(num_size)]

    ##3.
    #构造数据
    data=[]
    for i in range(num_size):
        ##数据量的大小，带宽的大小，服务器需要的计算时间ms，本机需要的计算时间ms
        ##服务器的处理能力是10^10，手机的处理能力是1.2*10^9
        adata=[L[i],D[i],w[i],(1/0.12)*w[i]]
        data.append(adata)
    Utils.save("data.npy",data)
    print(Utils.load("data.npy"))  #[[75.0, 100, 0.441], [85.0, 40, 0.27965], [100.0, 60, 0.765]]
    print(data)
    print("生成业务的长度为：",len(data))

