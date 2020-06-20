import numpy as np
import  random
from random import choice
import Email
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers import Dropout
from keras import initializers
import matplotlib.pyplot as plt
import Utils
import  copy
import random
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from scipy import stats
from keras.models import load_model
import yichuan
import data as Data
random.seed(10)
import sys

hidden_dims=40 #神经网络的总的隐藏神经元个数
# hidden_dims2=3000  #神经网络的总的隐藏神经元个数
B=100    #总的带宽限制
pici_size=4
def get_data():
    """
    处理输入的生成的随机的数据，随机产生的数据数量为3-10个
    :return: 一个二维的数组[[a,b,c,d]...]
    """
    data = np.matrix.tolist(Utils.load("data.npy"))
    # print("长度",len(data))
    Out_data=[]  #[[任务1,任务2.。。]，[任务1,任务2.。。]]
    index=0  #第几个data数据
    size=[]  #data的数量  #[5个任务,6个任务]自己查看可以用
    for i in range(int(Data.num_size/10)):
        data_num=round(random.random()*7+3)  #随机的数据上传的数量3-10个
        inner_data=[]
        for j in range(data_num):
            size.append(data_num)
            inner_data.append(data[index])
            index=index+1
        Out_data.append(inner_data)
    # print(Out_data)
    # print(size)
    for adata in Out_data:
        while(len(adata)<10):
            adata.append([0,0,0,0])
    return Out_data  #[[[39,40,88],[48,60,98]....[49,100,88]],...]

def get_input_data(Out_data):
    """
    输入很多原始随机生成的数据
    :param Out_data:
    :return:均值等等神经网络需要输入的数据
    """
    new_out_data = []
    for data in Out_data:
        inner_new_out_data=[]
        lie1=[x[0] for x in data]
        inner_new_out_data.append(round(np.mean(lie1),3))#数据量的列  均值
        inner_new_out_data.append(round(np.median(lie1),3) )#中位数
        inner_new_out_data.append(round(np.std(lie1),3)) ##标准差
        inner_new_out_data.append(round(stats.skew(lie1),3)) ##偏度：
        inner_new_out_data.append(round(stats.kurtosis(lie1),3)) ##峰度：

        lie2 = [x[2] for x in data]
        inner_new_out_data.append(round(np.mean(lie2),3))  # 计算量的列  均值
        inner_new_out_data.append(round(np.median(lie2),3))  # 中位数
        inner_new_out_data.append(round(np.std(lie2),3))  ##标准差
        inner_new_out_data.append(round(stats.skew(lie2),3))  ##偏度：
        inner_new_out_data.append(round(stats.kurtosis(lie2),3))  ##峰度


         ##  带宽40/60/100
        for x in data:
            if(x[1]==40):
                inner_new_out_data.append(1)
            elif(x[1]==60):
                inner_new_out_data.append(2)
            elif(x[1]==100):
                inner_new_out_data.append(3)
            else:
                inner_new_out_data.append(0)

        ##每个任务和最大任务的比较
        lie4 = [x[0] for x in data]
        max=np.max(lie4)
        for a in lie4:
            inner_new_out_data.append(round(a/max,3))

        #每个任务计算量和最大任务的比较
        lie5 = [x[2] for x in data]
        max = np.max(lie5)
        for a in lie5:
            inner_new_out_data.append(round(a/max,3) )

        # print(inner_new_out_data)
        ##队列的长度！！！
        new_out_data.append(inner_new_out_data)
    return new_out_data

def build_net_and_learn(xunlianflag,biaozhunhua_flag):

    inputs = Input(shape=(40,))
    activation='relu'
    arfa=1
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # a layer instance is callable on a tensor, and returns a tensor
    output_1 = Dense(hidden_dims*16*arfa, activation=activation,kernel_initializer=kernel_initializer)(inputs)
    output_1 = Dropout(0.05)(output_1)

    output_4 = Dense(hidden_dims*32*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_1)
    # output_21 = Dropout(0.05)(output_21)

    # output_22 = Dense(hidden_dims*8, activation=activation,kernel_initializer=kernel_initializer)(output_21)
    # # output_22 = Dropout(0.05)(output_22)
    #
    # output_23 = Dense(hidden_dims*16, activation=activation,kernel_initializer=kernel_initializer)(output_22)
    # output_23=Dropout(0.05)(output_23)
    #
    # output_2_ = Dense(hidden_dims*32, activation=activation,kernel_initializer=kernel_initializer)(output_23)
    # # output_2_ = Dropout(0.05)(output_2_)
    #
    # output_3 = Dense(hidden_dims*64, activation=activation,kernel_initializer=kernel_initializer)(output_2_)
    # # output_3 = Dropout(0.05)(output_3)
    #
    # output_4 = Dense(hidden_dims*64, activation=activation,kernel_initializer=kernel_initializer)(output_3) #隐藏层
    # # output_4=Dropout(0.05)(output_4)

    output_5 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第1个头
    output_5_2 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5) #第1个头
    output_5_3 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_2) #第1个头
    output_5_3 = Dense(hidden_dims*128*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_3) #第1个头
    output_5_3 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_3) #第1个头
    output_5_4 = Dense(hidden_dims*32*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_3) #第1个头
    output_5_5 = Dense(hidden_dims*16*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_4) #第1个头
    output_5_6 = Dense(hidden_dims*8*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_5) #第1个头
    output_5_7 = Dense(hidden_dims*4*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_6) #第1个头
    output_6 = Dense(hidden_dims*2*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_5_7)

    output_7 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第2个头
    output_7_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7)
    output_7_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_1)
    output_7_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_2)
    output_7_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_2)
    output_7_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_2)
    output_7_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)( output_7_3)
    output_7_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)( output_7_4)
    output_7_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_5)
    output_8 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_7_6)


    output_9 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第3个头
    output_9_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9)
    output_9_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_1)
    output_9_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_2)
    output_9_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_2)
    output_9_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_2)
    output_9_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_3)
    output_9_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_4)
    output_9_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_5)
    output_10 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_9_6)

    output_11 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第4个头
    output_11_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11)
    output_11_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_1)
    output_11_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_2)
    output_11_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_2)
    output_11_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_2)
    output_11_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_3)
    output_11_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_4)
    output_11_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_5)
    output_12 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_11_6)


    output_13 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第5个头
    output_13_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13)
    output_13_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_1)
    output_13_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_2)
    output_13_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_2)
    output_13_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_2)
    output_13_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_3)
    output_13_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_4)
    output_13_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_5)
    output_14 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_13_6)

    output_15 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第6个头
    output_15_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15)
    output_15_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_1)
    output_15_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_2)
    output_15_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_2)
    output_15_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_2)
    output_15_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_3)
    output_15_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_4)
    output_15_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_5)
    output_16 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_15_6)

    output_17 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第7个头
    output_17_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17)
    output_17_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_1)
    output_17_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_2)
    output_17_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_2)
    output_17_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_2)
    output_17_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_3)
    output_17_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_4)
    output_17_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_5)
    output_18 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_17_6)

    output_19 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第8个头
    output_19_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19)
    output_19_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_1)
    output_19_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_2)
    output_19_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_2)
    output_19_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_2)
    output_19_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_3)
    output_19_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_4)
    output_19_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_5)
    output_20 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_19_6)

    output_21 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第9个头
    output_21_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21)
    output_21_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_1)
    output_21_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_2)
    output_21_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_2)
    output_21_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_2)
    output_21_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_3)
    output_21_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_4)
    output_21_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_5)
    output_22 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_21_6)

    output_23 = Dense(hidden_dims*64*arfa, activation=activation,kernel_initializer=kernel_initializer)(output_4) #第10个头
    output_23_1 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23)
    output_23_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_1)
    output_23_2 = Dense(hidden_dims * 128*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_2)
    output_23_2 = Dense(hidden_dims * 64*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_2)
    output_23_3 = Dense(hidden_dims * 32*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_2)
    output_23_4 = Dense(hidden_dims * 16*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_3)
    output_23_5 = Dense(hidden_dims * 8*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_4)
    output_23_6 = Dense(hidden_dims * 4*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_5)
    output_24 = Dense(hidden_dims * 2*arfa, activation=activation, kernel_initializer=kernel_initializer)(output_23_6)


    #3批次，一个空批次
    predictions1 = Dense(pici_size,activation='softmax',name='output1')(output_6)
    predictions2 = Dense(pici_size,activation='softmax',name='output2')(output_8)
    predictions3 = Dense(pici_size,activation='softmax',name='output3')(output_10)
    predictions4 = Dense(pici_size,activation='softmax',name='output4')(output_12)
    predictions5 = Dense(pici_size,activation='softmax',name='output5')(output_14)
    predictions6 = Dense(pici_size,activation='softmax',name='output6')(output_16)
    predictions7 = Dense(pici_size,activation='softmax',name='output7')(output_18)
    predictions8 = Dense(pici_size,activation='softmax',name='output8')(output_20)
    predictions9 = Dense(pici_size,activation='softmax',name='output9')(output_22)
    predictions10 = Dense(pici_size,activation='softmax',name='output10')(output_24)

    model = Model(inputs=inputs, outputs=[predictions1, predictions2,
            predictions3,predictions4,predictions5,predictions6,
            predictions7, predictions8, predictions9, predictions10])
    model.compile(optimizer='sgd',
              loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                    'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  metrics=['accuracy'],
              )
    ##测试模块
    data=get_input_data(get_data())
    if (biaozhunhua_flag):
        zuida,zuixiao,data=biaozhunhua(data)
    X=[]
    # print(len(data))
    for i in range(int(len(data)*0.8)):
        # if(i==0):
        #     print(data[i])
        X.append(data[i])
    X=np.array(X)  #一个一维数组 np格式
    # Y1=np.array([[1,0,0,0]for i in range(5)]) #训练集的数据
    # Y2=np.array([[0,0,1,0]for i in range(5)])
    # Y3=np.array([[0,0,1,0]for i in range(5)])
    # Y4=np.array([[0,1,0,0]for i in range(5)])
    # Y5=np.array([[0,1,0,0]for i in range(5)])
    # Y6=np.array([[0,0,0,1]for i in range(5)])
    # Y7=np.array([[0,0,0,1]for i in range(5)])
    # Y8=np.array([[0,0,1,0]for i in range(5)])
    # Y9=np.array([[0,0,0,1]for i in range(5)])
    # Y10=np.array([[0,0,0,1]for i in range(5)])


    # split = int(len(X_old) * 0.8)
    # ##删除不好的数据
    # xunlianflag=[]
    # X=[]
    # yichuan_xunlian_time = Utils.load("xunliantime.npy")
    # yichuan_xunlian_time = yichuan_xunlian_time[0:split]
    # random_xunlian_time = Utils.load("random_xunlian80.npy")
    # for i in range(len(yichuan_xunlian_time)):
    #     if yichuan_xunlian_time[i]<random_xunlian_time[i]:
    #         xunlianflag.append(xunlianflag_old[i])
    #         X.append(X_old[i])
    # xunlianflag=np.array(xunlianflag)
    # X=np.array(X)

    Y1=[xunlianflag[i][0:4] for i in range(len(xunlianflag))]
    Y2=[xunlianflag[i][4:8] for i in range(len(xunlianflag))]
    Y3=[xunlianflag[i][8:12] for i in range(len(xunlianflag))]
    Y4=[xunlianflag[i][12:16] for i in range(len(xunlianflag))]
    Y5=[xunlianflag[i][16:20] for i in range(len(xunlianflag))]
    Y6=[xunlianflag[i][20:24] for i in range(len(xunlianflag))]
    Y7=[xunlianflag[i][24:28] for i in range(len(xunlianflag))]
    Y8=[xunlianflag[i][28:32] for i in range(len(xunlianflag))]
    Y9=[xunlianflag[i][32:36] for i in range(len(xunlianflag))]
    Y10=[xunlianflag[i][36:40] for i in range(len(xunlianflag))]
    ##开始训练
    print('Training -----------')
    history =model.fit(X, [Y1, Y2,Y3, Y4,Y5, Y6,Y7, Y8,Y9, Y10],
              epochs=100, batch_size=32)
    # plot_x=[]
    # plot_y=[]
    # for step in range(1000):
    #     cost = model.train_on_batch(X, [Y1, Y2,Y3, Y4,Y5, Y6,Y7, Y8,Y9, Y10])  # 训练的数据的输入，输出。。Keras有很多开始训练的函数，这里用train_on_batch（）
    #     # cost = model.train_on_batch(X, [Y2]) # Keras有很aiyaya多开始训练的函数，这里用train_on_batch（）
    #     if step % 10 == 0:
    #         plot_x.append(step)
    #         plot_y.append(cost[0])
    #         print('train cost: ',cost)
    #     # if(step!=0 and step%100==0):
    #     #     # 画图
    # plt.plot(plot_x, plot_y)
    # plt.savefig("cost.png")
    # plt.show()
    #保存训练好的模型 save model to single file
    ## 可视化
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    ##训练可视化
    # 绘制训练 & 验证的准确率值
    history_dict = history.history
    # print(history_dict.keys())

    # accuracy=history.history['output1_accuracy']+history.history['output2_accuracy']+history.history['output3_accuracy']+history.history['output4_accuracy']+history.history['output5_accuracy']+history.history['output6_accuracy']+history.history['output7_accuracy']+history.history['output8_accuracy']+history.history['output9_accuracy']+history.history['output10_accuracy']
    # print(accuracy)
    # print(type(accuracy[0]))

    plt.figure()
    plt.plot(history.history['output1_accuracy'])
    plt.title('Train Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('ca Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig("accuracy.png")
    plt.show()

    # 绘制训练 & 验证的损失值

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Train Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig("loss.png")
    plt.show()
    #
    model.save('model.h5')
    ##处理输出数据
    # process_pre_data()

    #计算总的使用时间
    if(biaozhunhua_flag):
        Utils.save("stand_max.npy", np.array([zuida]))
        Utils.save("stand_min.npy", np.array([zuixiao]))
def process_pre_data(model,adata,biaozhunhua_flag):
    """
    处理预测的数据结果：概率矩阵
    :param adata: 预测的数据集
    :return: 概率矩阵，以及预测了几个数据
    """
    # adata = np.array([[67.1, 85.0, 49.272, -0.232, -1.266, 0.409, 0.353, 0.396, 0.964, 0.241, 2, 1, 1, 3,
    #                    1, 3, 3, 0, 0, 0, 0.617, 0.603, 1.0, 0.603, 0.376, 0.879, 0.681, 0.0, 0.0, 0.0,
    #                    65.833, 64.319, 106.695, 64.319, 40.105, 93.831, 72.643, 0.0, 0.0, 0.0],
    #                   # [67.1, 85.0, 49.272, -0.232, -1.266, 0.409, 0.353, 0.396, 0.964, 0.241, 2, 1, 1, 3,
    #                   #  1, 3, 3, 0, 0, 0, 0.617, 0.603, 1.0, 0.603, 0.376, 0.879, 0.681, 0.0, 0.0, 0.0,
    #                   #  65.833, 64.319, 106.695, 64.319, 40.105, 93.831, 72.643, 0.0, 0.0, 0.0]
    #                   ])
    adata=np.array(adata)
    ####标准化
    if(biaozhunhua_flag):
        zuida = Utils.load("stand_max.npy")[0]
        zuixiao = Utils.load("stand_min.npy")[0]
        for i in range(len(adata)):
            for j in range(len(adata[0])):
                adata[i][j]=(adata[i][j]-zuixiao[j])/(zuida[j]-zuixiao[j])
    # print(adata)
    pred = model.predict(adata)  # 测试集的输入
    pred=np.array(pred)
    #np变为列表,并修改神经网络输出的值
    pred_data=[]
    for i in pred:
        pred_data.append(np.matrix.tolist(i))
    # print(pred_data)

    #  把三维数组的数据转换为决策：
    ##[array([[9.9999750e-01, 2.3309938e-06, 2.3168498e-06, 3.3314373e-06]],
    #  dtype=float32), array([[9.9999762e-01, 1.9672477e-06, 2.0064174e-06, 2.7757067e-06]],
    #  dtype=float32), array([[9.9999726e-01, 2.5039528e-06, 1.5597728e-06, 2.4061369e-06]]
    #  。。。。。]
    #  10个头的输出转化为：
    #[[[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]],
    # [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]],
    # [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]],
    # [[1, 0, 0, 0], [1, 0, 0, 0]]]
    #
    #
    pre_data_size=pred.shape[1]   #神经网络输出的数据，看看预测几组数据
    probability_array = [[[0 for i in range(pred.shape[2])] for j in range(pred.shape[1])] for k in
                         range(pred.shape[0])]
    for i in range(len(pred_data)):
        for j in range(len(pred_data[i])):
            probability_array[i][j][pred_data[i][j].index(max(pred_data[i][j]))] = 1
    # print(probability_array)
    #输出的是 输出的矩阵，以及预测了几个数据

    return probability_array , pre_data_size

def cul_pici(probability_array,pre_data_size):
    """
    神经网络输出计算批次的决策
    :param probability_array: 输出的矩阵
    :param pre_data_size: 数据数量的大小
    :return: 可能是二维数组也可能是一维数组，取决于要计算的数据数目
    """
    # print(probability_array)
    # print(pre_data_size)
    data = []
    assert len(np.array(probability_array).shape)==3
    if (pre_data_size == 1):
        for i in range(len(probability_array)): #0-9
            for j in range(len(probability_array[i])):  #1个数据
                data.append(probability_array[i][j].index(max(probability_array[i][j])))
    else:
        for num in range(pre_data_size): #2个数据
            temp = []
            for i in range(len(probability_array)): #0-9
                temp.append(probability_array[i][num].index(max(probability_array[i][num])))
            data.append(temp)
    # print(data)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]批次的选择
    return data


def daikuan(adata,pici):
    """
    计算每个批次分配的数据，然后计算分配的带宽资源
    :param adata: 10个任务的原始情况[[a,b,c,d],...]
    :param pici: 10个任务选择批次列表[0,1,2,3,0,2,1,3,...]
    :return: 批次带宽的分配,输出每个任务对应的资源的分配
    """
    #构造出字典，存储字典
    pici_dict={i:[] for i in range(pici_size)}
    for xuanze_index in range(len(pici)):
        pici_dict[pici[xuanze_index]].append(xuanze_index)
    # print("任务的批次分配情况：",pici_dict)  #{0: [0], 1: [3, 4], 2: [1, 2, 7], 3: [5, 6, 8, 9]}
    pici_dict2=copy.deepcopy(pici_dict)  #用于返回
    # print("原始的10个完整任务:",adata)  #[[[ 87.       60.        1.32153]
                  # [ 85.       40.        0.42415]
                  # [141.       40.        0.78114]
                  # [ 85.      100.        0.54655]...10个
    ###对每个批次的数据进行计算
    dic={}#保存key为 第几个任务，value为资源的分配

    for key in pici_dict.keys():  #key为1 ，
        if key!=0:
            task_list=pici_dict.get(key)  #[3,4]
            # print(task_list)
            task=[]
            for i in range(len(task_list)):
                task.append(list(adata[0][task_list[i]]))  #[[38,100,67],[48,60,87]]
            # print(task)
            cul_resource(task_list,task,dic,B)
    # print(dic)
    return pici_dict2,dic
def cul_resource(task_list,task,dic,BB):
    if (len(task_list) == 1):
        if(task[0][1]<BB):
            dic[task_list[0]] =task[0][1]
        else:
            dic[task_list[0]] =BB
    else:
        num=len(task_list)  #本批次的任务数量
        temp=[0 for i in range(num)]  #临时的任务分配列表
        for i in range(num):
            temp[i]=BB*np.array(task)[i][0]/(sum(np.array(task)[:,0])+0.01 ) #按照数据量计算资源分配的带宽  自己的数据量/总的数据量*B
        flag=[1  if temp[j]>task[j][1] else 0 for j in range(num)]
        for i in range(len(flag)):
            if(flag[i]==1):
                dic[task_list[i]]=task[i][1]
                b=task[i][1]
                task_list.remove(task_list[i])
                task.remove(task[i])
                # flag.remove(flag[i])
                cul_resource(task_list,task,dic,BB-b)
                break
            else:
                dic[task_list[i]]=temp[i]


def jisuan(origin_adata,dic,bdic):
    """
    根据原始数据和字典，计算出每个任务的开始时间和结束的时间
    :param origin_adata:  原始的10个数据 [[, ,],[],..]
    :param dic:  任务字典   {0: [0], 1: [3, 4], 2: [1, 2, 7], 3: [5, 6, 8, 9]}
    :param bdic:  资源的带宽安排  {0: 60.0, 3: 100.0, 4: 40.0, 1: 40.0, 2: 40.0, 7: 0.0, 5: 100.0, 6: 90.0, 8: 0.0, 9: 0.0}
    :return:
    """
    time_dic={}  #存放10个任务对应的开始时间和结束时间, 本地任务计算计算结束后的时间，传输任务计算传输结束时间{0:[0,1],3:[2,4],4:{2,5}...}
    length=len(dic.keys()) # 批次数量
    max_time=0
    pici_end_time=[]
    for key in dic.keys(): # 批次
        arr=[]
        length=len(dic[key]) # 任务的数量
        if length==0:
            continue
        ##处理0批次的任务
        if key==0:
            for i in dic[0]:
                end_time=origin_adata[0][i][3]
                arr.append(end_time)
                time_dic[i]=[0,end_time]
            pici_end_time.append(max(arr))
        else:
            for i in range(length):
                atask=dic[key][i]  #找到批次的第一个任务
                daikuan_num=bdic[atask]  # 找到带宽
                end_time=max_time+origin_adata[0][atask][0]*100 / (daikuan_num*100+0.2)  #100为调整的系数，计算上传结束时间，数据量/（带宽*10）
                arr.append(end_time)
                time_dic[atask]=[max_time, end_time]  #保存每个任务的传输开始时间和结束时间
            max_time=max(arr)
            pici_end_time.append(max_time)
    # print("任务的开始与结束时间",time_dic)  #每个任务的开始时间与结束时间{0：[0,1.44],1:[2.7,4.88],...}
    # print("批次上传结束时间",pici_end_time)

    ##对字典进行排序
    time_dic=dict(sorted(time_dic.items(),key=lambda e:e[1][1]))
    ##服务器的队长以及计算
    server=0
    start_jisuan=0
    end_dic={}
    for key in time_dic.keys():
        if key not in dic[0]:
            if start_jisuan==0:
                start_jisuan=time_dic[key][1]
                server=origin_adata[0][key][2]
                end_dic[key]=start_jisuan+server
                continue
            data=time_dic[key]  #每个任务的具体开始时间和结束时间[0,44]
            jisuanli_=origin_adata[0][key][2] #这个任务需要的计算力
            server=max(server+jisuanli_-data[1]+start_jisuan,0) #队列长度88
            start_jisuan=data[1]
            end_dic[key] = start_jisuan + server
        else:
            end_dic[key]=time_dic[key][1]  ##任务传输结束就是总的结束时间
    time=sum(end_dic.values())
    energy=0
    temp1=10**(-26)*(1.2*10**9)**3.3
    temp2 = 1.1*10 ** (-15) * (10 ** 10)**2  ##  =计算周期*消耗能量/circle
    temp_p=10**6  #功率的值
    # temp_server=10**4
    bili=1*10**(-5)
    for key in dic.keys():
        renwu=dic[key]
        for arenwu in renwu:
            if key==0:
                ##本地计算的能量
                energy=energy+temp1*origin_adata[0][arenwu][3]*bili
            else:
                e1=temp2*origin_adata[0][arenwu][2]*bili
                e2=origin_adata[0][arenwu][0]*temp_p/(origin_adata[0][arenwu][1]*10+0.1)*bili
                energy=energy+e1+e2
    # print(time,energy*0.3*10**(-3))
    return time+energy

def biaozhunhua(a):
    a = np.array(a)
    zuida = [max(a[:, i]) for i in range(len(a[0]))]
    zuixiao = [min(a[:, i]) for i in range(len(a[0]))]
    d = []
    for i in range(len(a)):
        e = []
        for j in range(len(a[0])):
            e.append((a[i][j] - zuixiao[j]) / (zuida[j] - zuixiao[j]))
        d.append(e)
    return zuida,zuixiao,d




if __name__ == "__main__":
    try:
        ##训练数据和神经网络
        # yichuan.main()
        origin_data=get_data()  #10个一组数据,很多数据[[[28,60,38],..]...]
        # print(origin_data)
        train_data=get_input_data(origin_data)  #神经元的输入数据生成[[...],[...]..]
        train_flag=[]  #等待收集
        biaozhunhua_flag = True

        ##遗传算法生成的标签训练神经网络
        xunlianflag = Utils.load("xunlianflag.npy")
        # print(xunlianflag.shape)
        build_net_and_learn(xunlianflag,biaozhunhua_flag)  # 训练网络+保存
        if(biaozhunhua_flag):
            zuida=Utils.load("stand_max.npy")[0]
            zuixiao = Utils.load("stand_min.npy")[0]
        split = int(len(origin_data) * 0.8)
        ##预测遍历以后的真实数据

        #      ##预测
        #原始的一次的数据，计算神经网络输入的数据

        pre_data=origin_data[split:len(origin_data)]
        pre_time=[]
        model = load_model('model.h5')
        for i in range(len(pre_data)):
            pre_adata=[]
            pre_adata.append(pre_data[i])
            adata = get_input_data(pre_adata)  # [[输入的数据]]一条或者多条
            a, b = process_pre_data(model,adata,biaozhunhua_flag)  # a 表示输出的概率，b表示几个数据
            # print(a)
            pici = cul_pici(a, b)
            print("神经网络的任务的批次分配情况：", pici)  # [0,2,2,1,1,3,3,2,3,3]
            # adata = np.array([[67.1, 85.0, 49.272, -0.232, -1.266, 0.409, 0.353, 0.396, 0.964, 0.241, 2, 1, 1, 3,
            #                    1, 3, 3, 0, 0, 0, 0.617, 0.603, 1.0, 0.603, 0.376, 0.879, 0.681, 0.0, 0.0, 0.0,
            #                    65.833, 64.319, 106.695, 64.319, 40.105, 93.831, 72.643, 0.0, 0.0, 0.0],
            #                   # [67.1, 85.0, 49.272, -0.232, -1.266, 0.409, 0.353, 0.396, 0.964, 0.241, 2, 1, 1, 3,
            #                   #  1, 3, 3, 0, 0, 0, 0.617, 0.603, 1.0, 0.603, 0.376, 0.879, 0.681, 0.0, 0.0, 0.0,
            #                   #  65.833, 64.319, 106.695, 64.319, 40.105, 93.831, 72.643, 0.0, 0.0, 0.0]
            #                   ])
            ##针对adata进行带宽资源的分配
            dic, b_dic = daikuan(pre_adata, pici)
            print("每个批次任务的带宽分配情况：", b_dic)
            time_=jisuan(pre_adata, dic, b_dic)
            pre_time.append(time_)
        print(pre_time)

        ##画图 x:数据集    y:神经网络预测的时间和真实的时间差
        Utils.save("network_pre_time%20.npy",pre_time)
        pre_time=Utils.load("network_pre_time%20.npy")
        real_time=Utils.load("xunliantime.npy")
        real_time=real_time[split:real_time.shape[0]]
        xx=[i for i in range(pre_time.shape[0])]
        plt.plot(xx,abs(pre_time-real_time))

        random_pre_20=Utils.load("random_pre20.npy")
        plt.plot(xx,abs(random_pre_20-real_time))
        plt.legend(['yichuan','random'],loc='upper right')
        plt.savefig("pre.png")
        plt.show()
        print(np.matrix.tolist(real_time))
        print("随机匹配-遗传算法的误差矩阵",np.matrix.tolist(random_pre_20-real_time))
        print("平均预测误差：",(sum((abs(pre_time-real_time))/real_time)/len(pre_time)))
        print("随机平均预测误差：",(sum((abs(random_pre_20-real_time))/real_time)/len(real_time)))
    finally:
        # Email.main1()
        pass

    # origin_adata=np.array([[[87.0, 60.0, 1.32153], [85.0, 40.0, 0.42415], [141.0, 40.0, 0.78114], [85.0, 100.0, 0.54655], [53.0, 40.0, 0.19239], [124.0, 100.0, 0.28272], [96.0, 100.0, 0.5376], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    # nohup python -u inputdata.py > test.out 2>&1 &

