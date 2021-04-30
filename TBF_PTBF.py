import random
import math
import re
from HST_construct import HST_C


# 默认值
# task数量
lt = 100
# worker数量
lw = 500
# 方差
m = 100
# 方差
sd = 20
# 预定义点数量
con_nodes = 4
# 运行次数
time = 1
# 变量
matrixs = []
tasks = []
workers = []
HST_tree = []
max_dis = 0
HST_beta = 0
HST_c = 0               # 真实分支数HST_c 
HST_D = 0
HST_S = []
HST_leaf = 0
TBF_epsilon = 0.1
pu = []
True_S = []

def dis(i,j):
    """
    计算Euclid距离\n

    param i : 二维坐标下的点\n
    param j : 二维坐标下的点\n
    """
    return math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2))


# 获取i和j的最近公共祖先所在层数
def LCA_lvl(i, j):
    re = 0
    if i == j:
        return re
    while i!=j:
        re = re + 1
        i = int(i/(HST_c))
        j = int(j/(HST_c))
    return re

######准备工作######
def pre():
    ######数据######
    fo1 = open('predefined'+str(con_nodes)+'.txt',"r")
    matrixs = eval(fo1.readlines()[0])
    # matrixs = [
    #     {'x': 1,'y': 1},
    #     {'x': 2,'y': 3},
    #     {'x': 5,'y': 3},
    #     {'x': 4,'y': 4}
    # ]
    fo1.close()
    # print(len(matrixs))

    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
    test_data = fo.readlines()
    tasks = eval(test_data[0])
    workers = eval(test_data[1])
    fo.close()
    # workers = [
    #     {'x': 2,'y': 3,'epsilon': 0.01},
    #     {'x': 2,'y': 2,'epsilon': 0.1},
    #     {'x': 5,'y': 5,'epsilon': 0.2},
    # ]
    # tasks = [
    #     {'x': 1,'y': 0,'epsilon': 0.1},
    #     {'x': 1,'y': 3,'epsilon': 0.1},
    #     {'x': 6,'y': 2,'epsilon': 0.1},
    # ]
    # print(len(tasks))
    # print(len(workers))

    ######建树#######
    # random.shuffle(matrixs)
    Hst = HST_C(matrixs)
    HST_tree = Hst.algorithm_1()
    HST_beta = Hst.beta
    max_dis = Hst.maxD
    HST_c = Hst.c-1
    HST_D = Hst.D
    HST_S = Hst.S
    HST_leaf = pow(HST_c, HST_D)
    return HST_tree,HST_beta,max_dis,HST_c,HST_D,HST_S,HST_leaf,tasks,workers

# 随机游走，降低复杂度
def random_walk(leaf,pu):
    I_upward = 1
    level = 0
    p = 0
    node = leaf
    ori_node = leaf
    while(1):
        p = random.random()
        # print(p,pu[level])
        if p < pu[level]:
            I_upward = 1
        else:
            I_upward = 0
        if I_upward == 1:
            level = level+1
            ori_node = node
            node = int(node/HST_c)
        else:
            break
    
    # 结点保持不变，无扰动，直接返回
    if level == 0:
        # print(leaf,'扰动结果为',leaf)
        return leaf

    # 扰动
    # 从HST_c-1个结点中均匀随机选择，index对应S[]中的下标
    anc = []
    for i in range(HST_c):
        index = node*(HST_c)+i
        if index != ori_node:
            anc.append(index)
    s = random.choice(anc)
    node = s
    level -= 1

    # 从c个结点中选择
    while(level!=0):
        anc = []
        for i in range(HST_c):
            index = node*(HST_c)+i
            anc.append(index)
        s = random.choice(anc)
        node = s
        level -= 1
    # print(leaf,'扰动结果为',s)
    return s


# 为worker进行扰动，返回work扰动后的位置（用W'序列描述）
def peturbed(node,pu):
    # 在初始化的点集中找到最近的点
    shortest_node = 1
    shortest_dis = 1e5
    # print(node)
    for i in range(len(True_S)):
        
        tmp_dis = dis(node,True_S[i])
        if tmp_dis < shortest_dis:
            shortest_dis = tmp_dis
            #shortest_node是真实节点在树中的下标
            shortest_node = True_S[i]['id']
    # 调用算法3获得扰动结果
    perturbed_node = random_walk(shortest_node,pu)
    # print(node,'扰动结果为',perturbed_node)
    return perturbed_node

def cal_pro(epsilon):
    wt = [0 for i in range(HST_D+1)]
    tw = [0 for i in range(HST_D+1)]
    pu = [0 for i in range(HST_D+1)]
    WT = 1

    for i in range(HST_D+1):
        wt[i] = math.exp((4-pow(2,i+2))*epsilon)
    for i in range(HST_D):  # 0 -> D-1
        WT += pow((HST_c),i)*(HST_c-1)*wt[i+1]

    tw[0] = WT
    tw[1] = WT-1
    for i in range(HST_D-1):
        tw[i+2] = tw[i+1]-pow((HST_c),i)*(HST_c-1)*wt[i+1]
    for i in range(HST_D):
        pu[i] = tw[i+1]/tw[i]
    return pu

# 算法4
def TBF_assign(W,pu):
    MA = []
    for item in range(len(tasks)):
        perturbed_node = peturbed(tasks[item],pu)
        # print('task',item,'扰动结果为',perturbed_node)
        # 分配worker，在树的叶子节点中(也就是W')找到一个最近的点，从W'删除点，将这次匹配记录到M中
        lvl = HST_D
        match = W[0]
        for w in W:
            tmp = LCA_lvl(w['position'],perturbed_node)
            # print(w['position'],' ',perturbed_node,' 公共祖先：',tmp)
            if tmp < lvl:
                lvl = tmp
                match = w
        # print('最小公共祖先层数：',lvl)
        MA.append({
            't': item,
            'w': match['id']
        })
        # print('匹配结果：',MA[-1])
        W.remove(match)
    return MA

def PTBF_assign(W):
    pu = [0 for i in range(HST_D+1)]
    MA = []
    for item in range(len(tasks)):
        pu = cal_pro(tasks[item]['epsilon'])
        perturbed_node = peturbed(tasks[item],pu)
        # print('task',item,'扰动结果为',perturbed_node)
        # 分配worker，在树的叶子节点中(也就是W')找到一个最近的点，从W'删除点，将这次匹配记录到M中
        lvl = HST_D
        match = W[0]
        for w in W:
            tmp = LCA_lvl(w['position'],perturbed_node)
            # print(w['position'],' ',perturbed_node,' 公共祖先：',tmp)
            if tmp < lvl:
                lvl = tmp
                match = w
        # print('最小公共祖先层数：',lvl)
        MA.append({
            't': item,
            'w': match['id']
        })
        # print('匹配结果：',MA[-1])
        W.remove(match)
    return MA


def TBF(epsilon = 0.1):
    pu = [0 for i in range(HST_D+1)]
    pu = cal_pro(epsilon)
    print('TBF固定——每层向上走的概率：',pu)

    W_w = []
    # 将所有worker扰动
    for i in range(len(workers)):
        # 返回扰动的下标
        re = peturbed(workers[i],pu)
        tmp = {
            'id' : i,
            'position' : re
        }
        W_w.append(tmp)
    # print('worker扰动后的下标：',W_w)
    # 处理task
    MA = TBF_assign(W_w,pu)
    return MA

def PTBF():
    pu = [0 for i in range(HST_D+1)]
    W_w = []

    # 将所有worker扰动
    for i in range(len(workers)):
        # 计算个性隐私预算的
        pu = cal_pro(workers[i]['epsilon'])
        # print('worker',i,'每层向上走的概率：',pu)
        re = peturbed(workers[i],pu)
        tmp = {
            'id' : i,
            'position' : re
        }
        W_w.append(tmp)
    # print('worker扰动后的下标：',W_w)
    # 处理task
    MA = PTBF_assign(W_w)
    return MA


HST_tree,HST_beta,max_dis,HST_c,HST_D,HST_S,HST_leaf,tasks,workers = pre()
for i in range(len(HST_S[0])):
    if HST_S[0][i] != []:
        True_S.append({
            'x' : HST_S[0][i][0]['x'],
            'y' : HST_S[0][i][0]['y'],
            'id' : i 
        })
print("##########建树##########")
print('HST树的最高层：',HST_D)
print('beta = ',HST_beta)
print('最远距离：',max_dis)
print('分支数：',HST_c)
print('根节点：',HST_S[HST_D])
print('叶子节点数：',len(HST_S[0]))
print('叶子节点：',HST_S[0])
print('叶子节点中的真实节点：',True_S)

print("\n##########TBF算法##########")
TBF_MA = TBF(TBF_epsilon)
print("TBF匹配结果大小：",len(TBF_MA))
# print("TBF匹配结果：",TBF_MA)
TBF_total_distance = 0
for i in TBF_MA:
    TBF_total_distance += dis(tasks[i['t']],workers[i['w']])
print('TBF算法总距离：',TBF_total_distance)

print("\n##########PTBF算法##########")
PTBF_MA = PTBF()
print("PTBF匹配结果大小：",len(PTBF_MA))
# print("PTBF匹配结果：",PTBF_MA)
PTBF_total_distance = 0
for i in PTBF_MA:
    PTBF_total_distance += dis(tasks[i['t']],workers[i['w']])
print('PTBF算法总距离：',PTBF_total_distance)