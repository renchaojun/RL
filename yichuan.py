import math, random
import numpy as np
import inputdata
import Utils
import copy
class Population:
    # 种群的设计
    def __init__(self, size, chrom_size, cp, mp, gen_max):
        # 种群信息合
        self.individuals = []  # 个体集合
        self.fitness = []  # 个体适应度集
        self.guji = []  # 个体适应度集
        self.selector_probability = []  # 个体选择概率集合
        self.new_individuals = []  # 新一代个体集合


        self.size = size  # 种群所包含的个体数
        self.chromosome_size = chrom_size  # 个体的染色体长度
        self.crossover_probability = cp  # 个体之间的交叉概率
        self.mutation_probability = mp  # 个体之间的变异概率

        self.generation_max = gen_max  # 种群进化的最大世代数
        self.age = 0  # 种群当前所处世代
        self.output_size=4
        self.elitist = {'chromosome': [0 for i in range(self.output_size*10)], 'fitness': 0}  # 最佳个体的信息
        self.set=set()
        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        weizhi=0
        while(True):  #50
            arr=[0 for i in range(self.output_size*10)]
            # rand=[1,1,2,2,3,3,4,4]
            # rand.append(random.randint(1,4))
            # rand.append(random.randint(1,4))
            # np.random.shuffle(rand)
            for j in range(10):
                index=random.randint(2,4)+j*self.output_size
                arr[index-1]=1
            if hash(str(arr)) not in self.set:
                self.individuals.append(arr)  #[[0 1,0,0,1,0,0,0..40个]*50]
                self.new_individuals.append([0 for i in range(self.output_size*10)]) #[[0000 0000 00..40个]*50]
                self.fitness.append(0)  #[00 ...]50个0
                self.fitness.append(0)  #[00 ...]50个0
                # self.guji.append(0)  #[00 ...]50个0
                self.selector_probability.append(0) #个体选择概率[00 ...]50个0
                self.set.add(hash(str(arr)))
                weizhi=weizhi+1
            if weizhi==self.size:
                break

    def fitness_func(self, juece,origin_adata,adata):
        '''适应度函数，可以根据个体的两个染色体计算出该个体的适应度'''
        a=[]
        i=0
        while True:
            temp = []
            if (i + 4 <= len(juece)):
                temp.append(list(juece[i:i + 4]))
                i += 4
                a.append(temp)
            else:
                break
        pici = inputdata.cul_pici(a, 1)
        dic, b_dic = inputdata.daikuan(origin_adata, pici)
        target=inputdata.jisuan(origin_adata, dic, b_dic)
        return 1/(target+1)
    def evaluate(self,origin_adata,adata):
        '''用于评估种群中的个体集合 self.individuals 中各个个体的适应度'''
        sp = self.selector_probability
        self.individuals=self.individuals+self.new_individuals

        for i in range(len(self.individuals)):
            # print(i, len(self.fitness))
            self.fitness[i]= self.fitness_func(self.individuals[i],origin_adata,adata)


        temp=np.argsort(np.array(self.fitness))[::-1]
        self.fitness.sort()
        self.fitness.reverse()  #从大到小排序

        arr_temp=[]
        for j in temp:
            if max(self.individuals[j])>0:
                arr_temp.append(self.individuals[j])
                if(len(arr_temp)==self.size):
                    break
        self.individuals=arr_temp

        temp_fitness=copy.copy(self.fitness)
        temp_fitness=temp_fitness[0:self.size]
        ft_sum = sum(temp_fitness)
        for i in range(self.size):
            sp[i] = temp_fitness[i] / float(ft_sum)  # 得到各个个体的生存概率
        for i in range(1, self.size):
            sp[i] = sp[i] + sp[i - 1]  # 需要将个体的生存概率进行叠加，从而计算出各个个体的选择概率

    # 轮盘赌博机（选择）
    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i
        # size=len(self.individuals)
        # return random.randint(0,size*0.1)
    # def select(self):
    #
    #     sp = copy.deepcopy(self.selector_probability)
    #     sp.sort(reverse=True)
    #     split=int(len(sp)*0.2)
    #     a=sp[split]
    #     arr=[]
    #     for i in range(len(sp)):
    #         if(self.selector_probability[i]>=a):
    #             arr.append(i)
    #     return random.choice(arr)
    #
    # 交叉
    def cross(self, chrom1, chrom2):

        p = random.random()  # 随机概率
        if chrom1 != chrom2 and p < self.crossover_probability:
            split=random.randint(0+1,9-1)*4
            length=len(chrom2)
            q1=chrom1[0:split]
            q2=chrom2[0:split]
            h1=chrom1[split:length]
            h2=chrom2[split:length]
            (chrom1, chrom2) = (q1 + h2, q2 + h1)
        return (chrom1, chrom2)

    # 变异
    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(0, self.output_size-1)
            start=t*4
            end=(t+1)*4-1
            bianhuaweizhi=random.randint(start,end)
            for index in range(start,end+1):
                chrom[index]=0
            chrom[bianhuaweizhi]=1
             # ^ 按位异或运算符：当两对应的二进位相异时，结果为1
        return chrom
    # 进化过程
    def evolve(self,origin_adata,adata):
        indvs = self.individuals
        new_indvs = self.new_individuals
        # 计算适应度及选择概率
        self.evaluate(origin_adata,adata)
        # 进化操作
        i = 0
        while True:
            # 选择两个个体，进行交叉与变异，产生新的种群
            idv1 = self.select()
            idv2 = self.select()
            # 交叉
            (idv1_, idv2_) = (indvs[idv1], indvs[idv2])
            (idv1_new, idv2_new) = self.cross(idv1_, idv2_)
            # 变异
            (idv1_new, idv2_new) = (self.mutate(idv1_new), self.mutate(idv2_new))
            if hash(str(idv1_new)) not in self.set:
                new_indvs[i]= idv1_new # 将计算结果保存于新的个体集合self.new_individuals中
                self.set.add(hash(str(idv1_new)))
                i=i+1
            if i >= self.size:
                break
            if hash(str(idv2_new)) not in self.set:
                new_indvs[i]= idv2_new # 将计算结果保存于新的个体集合self.new_individuals中
                self.set.add(hash(str(idv2_new)))
                i=i+1
            # 判断进化过程是否结束
            if i >= self.size:
                break

        # self.new_individuals=new_indvs
        # 最佳个体保留
        # 如果在选择之前保留当前最佳个体，最终能收敛到全局最优解。
        self.reproduct_elitist()

        # # 更新换代：用种群进化生成的新个体集合 self.new_individuals 替换当前个体集合
        # for i in range(self.size):
        #     self.individuals[i] = self.new_individuals[i]

    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = -1
        for i in range(self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness']= self.fitness[i]
        if (j >= 0):
            self.elitist['chromosome'] = self.individuals[j]
            self.elitist['age'] = self.age

    def run(self,origin_adata,adata):
        '''根据种群最大进化世代数设定了一个循环。
        在循环过程中，调用 evolve 函数进行种群进化计算，并输出种群的每一代的个体适应度最大值、平均值和最小值。'''
        for i in range(self.generation_max):
            self.evolve(origin_adata,adata)
            # print(i, max(self.fitness), sum(self.fitness) / self.size, min(self.fitness))
            print(i, sum(self.fitness) / self.size, 1/max(self.fitness))
        print(self.elitist['chromosome'],1/self.elitist['fitness'])
        return self.elitist
def main_fun(pop,origin_adata):
    adata = inputdata.get_input_data(origin_adata)  # [[输入的数据]]一条或者多条
    return pop.run(origin_adata, adata)
    # 5.5336 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

def main():
    """
        功能：遗传算法得到训练数据集
        输入：一组数据
        输出：elitist['chromosome'] 方案
        保存在数组中，xunlianflag.np
        时间也保存在数组中， xunliantime.np
        """
    # origin_adata = np.array([[[87.0, 60.0, 1.32153], [85.0, 40.0, 0.42415], [141.0, 40.0, 0.78114],
    #                           [85.0, 100.0, 0.54655], [53.0, 40.0, 0.19239], [124.0, 100.0, 0.28272],
    #                           [96.0, 100.0, 0.5376], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    data = inputdata.get_data()
    xunlianflag = []
    xunliantime = []
    rand_match=[]
    for adata in data:
        origin_adata = []
        origin_adata.append(adata)
        # 种群的个体数量为 50，染色体长度为 25，交叉概率为 0.8，变异概率为 0.1,进化最大世代数为 150
        pop = Population(1000, 24, 0.8, 0.2, 10)
        dic = main_fun(pop,origin_adata)
        xunlianflag.append(dic['chromosome'])
        xunliantime.append( 1/dic['fitness'])
    print("查看标签的完成时间：", xunliantime)
    print("数据的长度", len(data))
    print("输出的标签：", dic['chromosome'])

    split = int(len(xunlianflag) * 0.8)
    Utils.save("xunlianflag.npy", xunlianflag[0:split])   #训练的80%的数据的决策情况
    Utils.save("yuceflag.npy", xunlianflag[split:len(xunlianflag)])  #预测的20%数据的决策请看
    Utils.save("xunliantime.npy", xunliantime)  #训练的决策的所有消耗的时间
if __name__ == '__main__':
    main()