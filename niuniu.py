import pandas as pd
import numpy as np
import os


def load_large_dta(fname):
    import sys

    reader = pd.read_stata(fname, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(100 * 1000)
        while len(chunk) > 0:
            df = df.append(chunk, ignore_index=True)
            chunk = reader.get_chunk(100 * 1000)
            print('.')
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print('\nloaded {} rows'.format(len(df)))

    return df

def deconde_str(string):
    """
    解码 dta文件防止 乱码
    """
    re = string.encode('latin-1').decode('utf-8')
    return re

def jishu(newdata):
    dic={}
    for i in newdata:
        if i not in dic:
            dic[i]=1
        else:
            dic[i]=dic[i]+1
    return dic
if __name__ == '__main__':
    ##########################
    #第三个数据库
    file = "/home/chaojun/桌面/chfs2015_master_city_20180504.dta"
    df=load_large_dta(file)
    hhid =df['hhid']
    swqt=df['swgt']
    alldic={}  #社取编号:[家庭编号,家庭编号...]
    for sheqvindex in range(len(swqt)):
        if swqt[sheqvindex] not in alldic:
            alldic[swqt[sheqvindex]]=[]
        alldic[swqt[sheqvindex]].append(hhid[sheqvindex])
    print(alldic)
    ##########################


    df_2002_path = "/home/chaojun/桌面/chfs2015_ind_20190404w.dta"
    # df=load_large_dta(df_2002_path).head(10)
    df = load_large_dta(df_2002_path)
    ###加载hhid和hhead
    hhid = df['hhid'].values
    sex_=df['a2003']
    hhead=df['hhead']
    xingbie={}
    for bian_id in range(len(hhid)):
        if hhead[bian_id]==1 :
            xingbie[hhid[bian_id]]=sex_[bian_id]

    row = df.shape[0]
     #教育水平
    data = df['a2012']
    newdata = []
    for i in data:
        if i in [1]:
            newdata.append(0)
        elif i in [2]:
            newdata.append(6)
        elif i in [3]:
            newdata.append(9)
        elif i in [4]:
            newdata.append(12)
        elif i in [5]:
            newdata.append(13)
        elif i in [6]:
            newdata.append(15)
        elif i in [7]:
            newdata.append(16)
        elif i in [8]:
            newdata.append(19)
        elif i in [9]:
            newdata.append(22)
        else:
            newdata.append(0)
    # newdata11 = np.array(newdata).reshape(row, 1)
    data = df['hhid'].values
    dic={}
    for i in range(len(data)):
        if data[i] not in dic:
            dic[data[i]]=[newdata[i]]
        else:
            dic[data[i]].append(newdata[i])
    for key in dic:
        list=dic[key]
        length=len(list)
        dic[key]=round(sum(list)/length,2)
    # print(dic)





    df_2002_path = "/home/chaojun/桌面/chfs2015_hh_20180601.dta"
    # df=load_large_dta(df_2002_path).head(10)
    df=load_large_dta(df_2002_path)
    # 'A4002a','A4002b','A4004a','A4005a','A4006a','A4007aa'
    # df.to_csv("sensor.csv", header=df.columns.values)

    # df=pd.read_csv("sensor.csv")
    print(df.columns.tolist())
    row=df.shape[0]
    col=df.shape[1]

    columns = ['家庭编号','关注度A4002a', '接触金融类课程A4002b', '利率计算A4004a', '通货膨胀预期A4005a', '投资风险选择A4006a', '投资风险了解程度A4007','jrzs1','所在社区平均水平','是否参与股票市场D3105b','风险厌恶a4003','风险偏好a4003',' 教育程度a2012','性别']
    #编号
    temp=df['hhid']
    bianhao=[]
    jiaoyuchengdu=[]
    xingbie2=[]
    for i in temp:
        bianhao.append(i)
        jiaoyuchengdu.append(dic[i])
        xingbie2.append(xingbie[i])
    bianhao = np.array(bianhao).reshape(row, 1)
    jiaoyuchengdu = np.array(jiaoyuchengdu).reshape(row, 1)
    xingbie2 = np.array(xingbie2).reshape(row, 1)

    #a4002a
    data=df['a4002a']
    print('数据量大小',data.shape)
    newdata1=[]
    for i in data:
        if i in [1,2,3]:
            newdata1.append(1)
        elif i in [4, 5]:
            newdata1.append(0)
        else:
            newdata1.append(0)
    print("关注度A4002a统计情况：",jishu(newdata1))
    newdata1 =np.array(newdata1).reshape(row,1)
    print(len(newdata1))

    #a4002b
    data = df['a4002b']
    newdata2 = []
    for i in data:
        if i in [1]:
            newdata2.append(1)
        elif i in [2]:
            newdata2.append(0)
        else:
            newdata2.append(0)
    print("接触金融类课程A4002b统计情况：", jishu(newdata2))
    newdata2 = np.array(newdata2).reshape(row, 1)

    # a4004
    data = df['a4004']
    newdata3 = []
    for i in data:
        if i in [1, 3, 4]:
            newdata3.append(0)
        elif i in [2]:
            newdata3.append(1)
        else:
            newdata3.append(0)
    print("利率计算A4004a统计情况：", jishu(newdata3))
    newdata3 = np.array(newdata3).reshape(row, 1)

    # a4005a
    data = df['a4005']
    newdata4 = []
    for i in data:
        if i in [2, 3, 4]:
            newdata4.append(0)
        elif i in [1]:
            newdata4.append(1)
        else:
            newdata4.append(0)
    # print(df['a1001'].mean(axis=0))
    print("通货膨胀预期A4005a统计情况：", jishu(newdata4))
    newdata4 = np.array(newdata4).reshape(row, 1)

    # a4006a
    data = df['a4006']
    newdata5 = []
    for i in data:
        if i in [2]:
            newdata5.append(1)
        elif i in [1]:
            newdata5.append(0)
        else:
            newdata5.append(0)
    print("投资风险选择A4006a统计情况：", jishu(newdata5))
    newdata5 = np.array(newdata5).reshape(row, 1)

    # a4007a
    data = df['a4007']
    newdata6 = []
    for i in data:
        if i in [1]:
            newdata6.append(1)
        elif i in [2,3,4,5]:
            newdata6.append(0)
        else:
            newdata6.append(0)
    print("投资风险了解程度A4007统计情况：", jishu(newdata6))
    newdata6 = np.array(newdata6).reshape(row, 1)

    #是否参与股票市场
    data = df['d3105b']
    data2 = df['d3101']
    newdata7 = []
    for i in range(len(data)):
        if data[i] in [1] and data2[i] in [1]:
            newdata7.append(1)
        else:
            newdata7.append(0)
    # print("是否参与股票市场D3105b统计情况：", jishu(newdata7))
    newdata7 = np.array(newdata7).reshape(row, 1)



    # # father教育
    # data = df['a2032_1']
    # newdata9 = []
    # for i in data:
    #     if i in [1]:
    #         newdata9.append(0)
    #     elif i in [2]:
    #         newdata9.append(6)
    #     elif i in [3]:
    #         newdata9.append(9)
    #     elif i in [4]:
    #         newdata9.append(12)
    #     elif i in [5]:
    #         newdata9.append(13)
    #     elif i in [6]:
    #         newdata9.append(15)
    #     elif i in [7]:
    #         newdata9.append(16)
    #     elif i in [8]:
    #         newdata9.append(19)
    #     elif i in [9]:
    #         newdata9.append(22)
    #     else:
    #         newdata9.append(i)
    # # print("父亲教育程度a2032-1统计情况：", jishu(newdata9))
    # newdata9 = np.array(newdata9).reshape(row, 1)
    #
    # # mother教育
    # data = df['a2032_2']
    # newdata10 = []
    # for i in data:
    #     if i in [1]:
    #         newdata10.append(0)
    #     elif i in [2]:
    #         newdata10.append(6)
    #     elif i in [3]:
    #         newdata10.append(9)
    #     elif i in [4]:
    #         newdata10.append(12)
    #     elif i in [5]:
    #         newdata10.append(13)
    #     elif i in [6]:
    #         newdata10.append(15)
    #     elif i in [7]:
    #         newdata10.append(16)
    #     elif i in [8]:
    #         newdata10.append(19)
    #     elif i in [9]:
    #         newdata10.append(22)
    #     else:
    #         newdata10.append(i)
    # # print("母亲教育程度a2032-2统计情况：", jishu(newdata10))
    # newdata10 = np.array(newdata10).reshape(row, 1)


    # 风险态度
    data = df['a4003']
    newdata12 = []
    newdata13 = []
    for i in data:
        if i in [1,2,3,6]:
            newdata12.append(0)
        elif i in [4, 5]:
            newdata12.append(1)
        else:
            newdata12.append(None)
    for i in data:
        if i in [1,2]:
            newdata13.append(1)
        elif i in [3,4, 5,6]:
            newdata13.append(0)
        else:
            newdata13.append(None)
    # print("风险厌恶a4003统计情况：", jishu(newdata12))
    # print("风险偏好a4003统计情况：", jishu(newdata13))
    newdata12 = np.array(newdata12).reshape(row, 1)
    newdata13 = np.array(newdata13).reshape(row, 1)




    #第一部分的数据
    m = np.hstack((bianhao, newdata1, newdata2, newdata3, newdata4, newdata5, newdata6)).tolist()
    newm=[]
    for row in m:
        num=0
        for adata in row:
            if adata=='1':
                num=num+1
        row.append(num)
        newm.append(row)



    ##社取编号：[分]
    dic2={}
    for key in alldic:
        if key not in dic2:
            dic2[key] = []
    for adata in newm:
        id=adata[0]
        fen=adata[-1]
        for key in alldic:  #社取
            if id in alldic[key]:  #家庭编号列表
                dic2[key].append(fen)  #社取的所有的分
    print(dic2)

    newline=[]
    for line in newm:
        id=line[0]
        fen = line[-1]
        for key in dic2:  #社取
            if id in alldic[key]:  #分
                if(len(dic2[key])==1):
                    line.append(sum(dic2[key]))
                else:
                    line.append((sum(dic2[key])-fen)/(len(dic2[key])-1))
                newline.append(line)


    newm = np.array(newline)
    print(newm.shape)
    print(newdata7.shape)
    newdata=np.hstack((newm,newdata7,newdata12,newdata13,jiaoyuchengdu,xingbie2))

    print(newdata)
    xinshuju= pd.DataFrame(np.array(newdata),columns=columns)
    # xinshuju.dropna(axis=0, how='any', inplace=True)
    # print(xinshuju.values)

    xinshuju.to_csv("processed_data.csv", header=xinshuju.columns.values)







