import random
import math
import re
from HST_construct import HST_C


# 默认值
# task数量
lt = 1000
# worker数量
lw = 5000
# 方差
m = 100
# 方差
sd = 20
# 预定义点数量
con_nodes = 50
# 运行次数
time = 1
# 变量
matrixs = []
tasks = []
workers = []
HST_tree = []
max_dis = 0
HST_beta = 0
# 真实分支数HST_c 
HST_c = 0
HST_D = 0
HST_leaf = 0
TBF_epsilon = 0.6
wt = []
tw = []
pu = []
WT = 0



######准备工作######
def pre():
    ######数据######
    fo1 = open('predefined'+str(con_nodes)+'.txt',"r")
    matrixs = eval(fo1.readlines()[0])
    fo1.close()
    print(len(matrixs))

    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
    test_data = fo.readlines()
    tasks = eval(test_data[0])
    workers = eval(test_data[1])
    fo.close()
    print(len(tasks))
    print(len(workers))

    ######建树#######
    random.shuffle(matrixs)
    Hst = HST_C(matrixs)
    HST_tree = Hst.algorithm_1()
    HST_beta = Hst.beta
    max_dis = Hst.maxD
    HST_c = Hst.c-1
    HST_D = Hst.D
    HST_leaf = pow(HST_c, HST_D)


def cal_pro(epsilon):
    wt = [0 for i in range(HST_D+1)]
    tw = [0 for i in range(HST_D+1)]
    pu = [0 for i in range(HST_D+1)]
    WT = 1

    for i in range(HST_D+1):
        wt[i] = math.exp((4-pow(2,i+2))*epsilon)
    for i in range(HST_D):  # 0 -> D-1
        WT += pow((HST_c),i)*(HST_c-1)*wt[i+1]
    print('wt向量：',wt)

    tw[0] = WT
    tw[1] = WT-1
    for i in range(HST_D-1):
        tw[i+2] = tw[i+1]-pow((HST_c),i)*(HST_c-1)*wt[i+1]
    for i in range(HST_D):
        pu[i] = tw[i+1]/tw[i]
    print('每层向上走的概率：',pu)


def TBF(epsilon = 0.6):
    cal_pro(epsilon)
    # print(pu)



pre()
TBF()