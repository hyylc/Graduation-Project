import random
import math
import re



# 默认值
# task数量
lt = 100
# worker数量
lw = 500
# 均值
m = 100
# 标准差
sd = 20
# 预定义点数量
con_nodes = 4
# 运行次数
time = 1
# 变量
tasks = []
workers = []
pu = []
HST_D = 0
HST_c = 0
True_S = []
# 块间距离
dis = [[0 for i in range(25)] for i in range(25)]
block_M = [[0 for i in range(25)] for i in range(25)]
for i in range(5):
    for j in range(5):
        for l in range(5):
            for n in range(5):
                dis[i*5+j][l*5+n] = max(i-l,l-i) + max(j-n,n-j)
# print(dis)

def node_dis(i,j):
    """
    计算Euclid距离\n

    param i : 二维坐标下的点\n
    param j : 二维坐标下的点\n
    """
    return math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2))

#任务和任务接收方集合
def pre():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
    test_data = fo.readlines()
    tasks = eval(test_data[0])
    workers = eval(test_data[1])
    fo.close()
    return tasks,workers

# 块间扰动
def perturb_block(node):
    alpha = 0
    sum = 0
    # 概率
    p = random.random() 
    # print(p)
    b = int(node['x']/40)+int(node['y']/40)*5
    # print('真实块号',b)
    for l in range(5):
        for n in range(5):
            alpha += pow(math.e,-(node['epsilon']/2)*dis[b][l*5+n])
    for l in range(5):
        for n in range(5):
            block_M[b][l*5+n] = pow(math.e,-(node['epsilon']/2)*dis[b][l*5+n])/alpha
            sum1 = sum + block_M[b][l*5+n]
            # 边计算概率，边得到扰动结果
            if p > sum and p <= sum1:
                return  l*5+n
            else:
                sum = sum1
    return 24
    

# 随机游走，降低复杂度
def random_walk(HST_c,leaf,pu):
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
    # print('第',level,'层结点:',node,'向下选择:',anc)
    s = random.choice(anc)
    node = s
    level -= 1

    # 从c个结点中选择
    while(level!=0):
        anc = []
        for i in range(HST_c):
            index = node*(HST_c)+i
            anc.append(index)
        # print('第',level,'层结点:',node,'向下选择:',anc)
        s = random.choice(anc)
        node = s
        level -= 1
    # print(leaf,'扰动结果为',s)
    return s


# 块内扰动，HST树扰动
def perturb_HST(HST_c,node,pu,True_S):
    # 在初始化的点集中找到最近的点
    shortest_node = 1
    shortest_dis = 1e5
    # print(node)
    for i in range(len(True_S)):
        tmp_dis = node_dis(node,True_S[i])
        if tmp_dis < shortest_dis:
            shortest_dis = tmp_dis
            #shortest_node是真实节点在树中的下标
            shortest_node = True_S[i]['id']
    # 调用算法3获得扰动结果
    # print(node,'最近的点为',shortest_node)
    perturbed_node = random_walk(HST_c,shortest_node,pu)
    # print(node,'扰动结果为',perturbed_node)
    return perturbed_node

def cal_pro(HST_D,HST_c,epsilon):
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
    # print('每层向上走的概率：',pu)
    return pu

# 根据二维空间平面的真实坐标node得到扰动结果：（块号，节点号）
def perturb(node):
    # 块间扰动，得到扰动块号
    p_b = perturb_block(node)
    # print('扰动块号',p_b)
    # 读取对应块的HST树信息
    fo = open(str(p_b)+"_HST.txt", "r")
    HST_info = eval(fo.readlines()[0])
    # print(HST_info)
    fo.close()
    HST_D = HST_info['HST_D']
    HST_c = HST_info['HST_c']
    True_S = HST_info['HST_true_S']
    pu = cal_pro(HST_D,HST_c,node['epsilon'])
    p_leaf = perturb_HST(HST_c,node,pu,True_S)
    # print('扰动叶子节点号',p_leaf)
    return p_b,p_leaf,HST_D,HST_c




# 根据块号和节点号，判断获取i和j的最近公共祖先所在层数
def LCA_lvl(HST_c,i, j):
    re = 0
    if i == j:
        return re
    while i!=j:
        re = re + 1
        i = int(i/(HST_c))
        j = int(j/(HST_c))
    return re

def Exp_TBF_assign(W):
    M = []
    for item in range(len(tasks)):
        re_b,re_leaf,HST_D,HST_c = perturb(tasks[item])
        tmp1 = -1
        tmp2 = 0
        lvl = HST_D
        d = 8
        for i in range(len(W)):
            if W[i]['block'] == re_b:
                tmp = LCA_lvl(HST_c,W[i]['leaf'],re_leaf)
                if tmp < lvl:
                    lvl = tmp
                    tmp1 = i
            else:
                tmp = dis[re_b][W[i]['block']]
                if tmp < d:
                    d = tmp
                    tmp2 = i
        if tmp1 != -1:
            M.append({
                't': item,
                'w': W[tmp1]['id']
            })
            W.remove(W[tmp1])
        else:
            M.append({
                't': item,
                'w': W[tmp2]['id']
            })
            W.remove(W[tmp2])
        # print('匹配结果：',M[-1])
    return M

def Exp_TBF():
    # 将所有worker扰动，再对task一个一个扰动，分配
    W_w = []
    for i in range(len(workers)):
        re_b,re_leaf,HST_D,HST_c = perturb(workers[i])
        W_w.append({
            'id' : i,
            'block' : re_b,
            'leaf' : re_leaf
        })
    MA = Exp_TBF_assign(W_w)
    # print('分配总数：',len(MA))
    total_distance = 0
    for i in MA:
        total_distance += node_dis(tasks[i['t']],workers[i['w']])
    # print('Exp_TBF算法总距离：',total_distance)
    return total_distance


T_size = [100,200,300,400,500]
W_size = [300,400,500,600,700]
mean = [50,75,100,125,150]
sigma = [10,15,20,25,30]


T_variety = []
W_variety = []
m_variety = []
s_variety = []
e_variety = []



# print('任务数量取100-500')
# for i in range(len(T_size)):
#     lt = T_size[i]
#     lw = 500
#     m = 100
#     sd = 20
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Exp_TBF()
#     distance = distance/20
#     print('20次运行结果的均值：Exp_TBF算法总距离：',distance)
#     T_variety.append(distance)

# print('任务接收方数量取300-700')
# for i in range(len(W_size)):
#     lt = 300
#     lw = W_size[i]
#     m = 100
#     sd = 20
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Exp_TBF()
#     distance = distance/20
#     print('20次运行结果的均值：Exp_TBF算法总距离：',distance)
#     W_variety.append(distance)

# print('均值取50-150')
# for i in range(len(mean)):
#     lt = 300
#     lw = 500
#     m = mean[i]
#     sd = 20
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Exp_TBF()
#     distance = distance/20
#     print('20次运行结果的均值：Exp_TBF算法总距离：',distance)
#     m_variety.append(distance)

# print('标准差取10-30')
# for i in range(len(sigma)):
#     lt = 300
#     lw = 500
#     m = 100
#     sd = sigma[i]
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Exp_TBF()
#     distance = distance/20
#     print('20次运行结果的均值：Exp_TBF算法总距离：',distance)
#     s_variety.append(distance)

# print('隐私预算取五个区间')
# for i in range(5):
#     lt = 300
#     lw = 500
#     m = 100
#     sd = 20
#     a = round(0.2+0.2*i,1)
#     b = 1.2
#     fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+"_"+str(a)+"_"+str(b)+".txt", "r")
#     test_data = fo.readlines()
#     tasks = eval(test_data[0])
#     workers = eval(test_data[1])
#     fo.close()
#     distance = 0
#     for times in range(20):
#         distance += Exp_TBF()
#     distance = distance/20
#     print(a,b)
#     print('20次运行结果的均值：Exp_TBF算法总距离：',distance)
#     e_variety.append(distance)

# # 写文件
# fo = open("T_vary.txt", "a+")
# fo.write(str(T_variety))
# fo.write('\n')
# fo.close()
# fo = open("W_vary.txt", "a+")
# fo.write(str(W_variety))
# fo.write('\n')
# fo.close()
# fo = open("m_vary.txt", "a+")
# fo.write(str(m_variety))
# fo.write('\n')
# fo.close()
# fo = open("s_vary.txt", "a+")
# fo.write(str(s_variety))
# fo.write('\n')
# fo.close()
# fo = open("e_vary.txt", "a+")
# fo.write(str(e_variety))
# fo.write('\n')
# fo.close()