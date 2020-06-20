import time
import numpy as np
import math, random
import numpy as np
import inputdata
import Utils
import copy

from concurrent.futures import ThreadPoolExecutor
import threading


def f(x):
    a = []
    while x != 0:
        a.append(x % 4)
        x = x // 4
    a.reverse()
    return "".join([str(d) for d in a])

def fitness_func(juece,origin_adata,adata):
    '''适应度函数，可以根据个体的两个染色体计算出该个体的适应度'''
    pici = inputdata.cul_pici([juece], 1)
    dic, b_dic = inputdata.daikuan(origin_adata, pici)
    xiaohao=inputdata.jisuan(origin_adata, dic, b_dic)
    return xiaohao

def test(data,num,xunlianflag,xunlianxiaohao):
    adata=data[num]
    print("%s threading is processed %s"%(threading.current_thread().name, adata))

    origin_adata = []
    origin_adata.append(adata)
    mintime = 100
    favjuece = None
    i = 0
    for juece in newarr:
        if(i%1000==0):
            print(threading.current_thread().name,i)
        i = i + 1
        adata = inputdata.get_input_data(origin_adata)
        time = fitness_func(juece, origin_adata, adata)
        if (time < mintime):
            mintime = time
            favjuece = juece
    xunlianxiaohao[i]=mintime
    xunlianflag[i]=favjuece

def test_result(future):
    print(future.result())


if __name__ == "__main__":

    threadPool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_")

    newarr = np.matrix.tolist(Utils.load("bianli.npy"))

    data = inputdata.get_data()
    xunlianflag = {}
    xunlianxiaohao = {}


    for i in range(len(data)):
        if i>2:
            break
        future = threadPool.submit(test, data,i,xunlianflag,xunlianxiaohao)


    threadPool.shutdown(wait=True)

    bianlixunlianflag=[]
    bianlixunlianxiaohao=[]
    for i in range(len(data)):
        bianlixunlianflag.append(xunlianflag[i])
        bianlixunlianxiaohao.append(xunlianxiaohao[i])
    Utils.save("bianlixunlianflag.npy",bianlixunlianflag)
    Utils.save("bianlixunlianxiaohao.npy",bianlixunlianxiaohao)
    print(bianlixunlianflag)
    print(bianlixunlianxiaohao)