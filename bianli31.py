import math, random
import numpy as np
import inputdata
import Utils
import copy

# for adata in data:
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
if __name__ == '__main__':
    # end=1500000
    # data=[]
    #
    # newarr=[]
    # for i in range(end):
    #     m=f(i)
    #     if(len(m)>10):
    #         break
    #     temp=list(m.zfill(10))
    #     ##对temp进行处理 ['3', '3', '2', '2', '3', '3', '2', '3', '0', '1']
    #     arr = [[0] * 4] * 10
    #     for i in range(len(temp)):
    #         if(temp[i]=='0'):
    #             arr[i][0]=1
    #         elif(temp[i]=='1'):
    #             arr[i][1] = 1
    #         elif(temp[i]=='2'):
    #             arr[i][2] = 1
    #         elif(temp[i] == '3'):
    #             arr[i][3] = 1
    #     newarr.append(arr)
    # print(newarr)
    # Utils.save("bianli.npy",newarr) #三维数组

    newarr=np.matrix.tolist(Utils.load("bianli.npy"))
    data = inputdata.get_data()
    xunlianflag = []
    xunliantime = []
    rand_match = []
    num = 0
    len=len(data)
    for adata in data[int(len*30/40):int(len*31/40)]:
        origin_adata = []
        origin_adata.append(adata)
        mintime = 1000000
        favjuece = None
        i = 0
        num = num + 1
        for juece in newarr:
            if (i % 1000 == 0):
                print(juece, num, "->", i)
            i = i + 1
            adata = inputdata.get_input_data(origin_adata)
            time = fitness_func(juece, origin_adata, adata)
            if (time < mintime):
                mintime = time
                favjuece = juece
                # print(favjuece)
                # print(mintime)
        xunlianflag.append(favjuece)
        xunliantime.append(mintime)
    Utils.save("bianlixunlianflag31.npy", xunlianflag)
    Utils.save("bianlixunlianxiaohao31.npy", xunliantime)
    print(xunliantime)