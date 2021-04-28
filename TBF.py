import random
import math
import re

# 树类
def HST(r):
    """
    param r : 新的节点\n
    """
    return [r]

def dis(i,j):
    """
    计算Euclid距离\n

    param i : 二维坐标下的点\n
    param j : 二维坐标下的点\n
    """
    return math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2))

def max_dis(V):
    """
    求最远距离\n
    param v : 点集\n
    """
    d = 0.0
    for i in V:
        for j in V:
            d = max(d, dis(i, j))
    return d


def construct(tree, i):
    """
    构建HST树\n
    :param tree: 待划分的父亲结点\n
    :param i: 新结点的层数\n
    """
    # 一棵只有根结点的树
    if i < 0:
        return
    T = tree[0]
    r[i] = beta*pow(2,i)
    global c
    for vertex_i in PI:
        if len(T) == 0:
            return
        temp = []
        U = []
        new_T = []
        for vertex_j in metrixs:
            if dis(vertex_i,vertex_j) <= r[i]:
                temp.append(vertex_j)
        # temp 和 T 的交集
        for vertex in temp:
            if vertex in T:
                U.append(vertex)
        # T - U
        for vertex in T:
            if vertex not in U:
                new_T.append(vertex)
        # U 不为空，新建结点，递归构建树
        if len(U) != 0:
            HST_U = HST(U)
            # print('距离点', vertex_i, r[i], '的点有', U)
            tree.append(HST_U)
            # S[i].append(HST_U)
            #这个不应该在这里，应该在add fake nodes之后 S[i].append(U)
            construct(HST_U, i-1)
            T = new_T
            # print('当前T为 ',T)

    
    c = max(c, len(tree))
    print('当前分支数',c)
        

def add_fake_nodes(HST_tree, level):
    """
    添加fake nodes\n
    :param HST_tree: 待补充fake nodes的树，第一个元素是根节点\n
    :param level: 新结点的层数\n
    """
    if level < 0:
        return
    l = len(HST_tree)
    # 对应数量 c-l 的fake nodes
    for i in range(c-l):
        HST_tree.append([])
    # 遍历HST_tree的子节点
    for i in range(len(HST_tree)-1):
        add_fake_nodes(HST_tree[i+1], level-1)


def print_tree(HST_tree, level):
    """
    先序遍历树\n
    构建S[D]\n
    :param HST_tree: 树\n
    :param level: 当前根结点的层数\n
    """
    if level < 0:
        return
    if len(HST_tree) != 0:
        print(HST_tree[0])
    elif level == 0:
        print(HST_tree)
    for i in range(len(HST_tree)-1):
        print_tree(HST_tree[i+1], level-1)

def get_S(HST_tree, level):
    """
    构建S[D]\n
    :param HST_tree: 树\n
    :param level: 当前根结点的层数\n
    """
    if level < 0:
        return
    if len(HST_tree) != 0:
        try:
            S[level].append(HST_tree[0])
        except:
            print(HST_tree)
    elif level == 0:
        S[level].append(HST_tree)
    for i in range(len(HST_tree)-1):
        get_S(HST_tree[i+1], level-1)

def print_leaf(HST_tree, level):
    if level == 0:
        print(HST_tree)
        return
    for i in range(len(HST_tree)-1):
        print_leaf(HST_tree[i+1], level-1)

# 每两个结点之间的最近公共祖先所在层数
def LCA_level(level, start, end):
    if start == end:
        LCA[start][end] = 0
        return
    for i in range(c-1):
        for j in range(c-1):
            if i != j:
                for k in range(pow(c-1, level-1)):
                    for l in range(pow(c-1, level-1)):
                        LCA[start+i*pow(c-1, level-1)+k][start+j*pow(c-1, level-1)+l] = level
    for i in range(c-1):
        LCA_level(level-1, start+i*pow(c-1, level-1), start+(i+1)*pow(c-1, level-1)-1)

def print_format(M):
    for i in range(len(M)):
        print(M[i])


# 算法1
def algorithm_1(V):
    """
    :param V: 原空间上的点的集合\n
    """
    # 构造HST树
    HST_tree = HST(V)
    construct(HST_tree, D-1)
    add_fake_nodes(HST_tree, D-1)
    # print(HST_tree)
    return HST_tree


# 算法2 根据概率M[x][a]扰动


# 算法3 等价算法2效果，降低复杂度（根据S[]）
def algorithm_3(leaf):
    I_upward = 1
    level = 0
    p = 0
    node = leaf
    ori_node = leaf
    while(1):
        p = round(random.random(),3)
        # print(p,pu[level])
        if p < pu[level]:
            I_upward = 1
        else:
            I_upward = 0
        if I_upward == 1:
            level = level+1
            ori_node = node
            node = int(node/(c-1))
        else:
            break
    
    # 结点保持不变，无扰动，直接返回
    if level == 0:
        return leaf

    # 扰动
    # 从c-1个结点中均匀随机选择，index对应S[]中的下标
    anc = []
    for i in range(c-1):
        index = node*(c-1)+i
        if index != ori_node:
            anc.append(index)
    # print('第',level,'层结点:',node,'向下选择:',anc)
    s = random.choice(anc)
    node = s
    level -= 1

    # 从c个结点中选择
    while(level!=0):
        anc = []
        for i in range(c-1):
            index = node*(c-1)+i
            anc.append(index)
        # print('第',level,'层结点:',node,'向下选择:',anc)
        s = random.choice(anc)
        node = s
        level -= 1
    return s


# 为worker进行扰动，返回work扰动后的位置（用W'序列描述）
def worker_peturbed(node):
    '''
    
    '''
    # 在初始化的点集中找到最近的点
    shortest_node = 1
    shortest_dis = 1e5 
    # print(S[0])
    for i in range(len(S[0])):
        if S[0][i] != []:
            tmp_dis = dis(node,S[0][i][0])
            if tmp_dis < shortest_dis:
                shortest_dis = tmp_dis
                shortest_node = i
    # 调用算法3获得扰动结果
    # print(node,'最近的点',shortest_node)
    # 调用算法3获得扰动结果
    perturbed_node = algorithm_3(shortest_node)
    # print(node,'扰动结果为',perturbed_node)
    return perturbed_node
    

# 算法4
def algorithm_4(node_list):
    '''
    '''
    for item in range(len(node_list)):
        # 扰动的步骤和worker是完全一样
        # 在初始化的点集中找到最近的点
        shortest_node = 1
        shortest_dis = 1e5 
        # 和叶子节点比较
        for i in range(len(S[0])):
            if S[0][i] != []:
                tmp_dis = dis(node_list[item],S[0][i][0])
                if tmp_dis < shortest_dis:
                    shortest_dis = tmp_dis
                    shortest_node = i
        # 调用算法3获得扰动结果
        # print('task',item,'最近的点',shortest_node)
        # 调用算法3获得扰动结果
        perturbed_node = algorithm_3(shortest_node)
        # print('task',item,'扰动结果为',perturbed_node)
        # 分配worker，在树的叶子节点中(也就是W')找到一个最近的点，从W'删除点，将这次匹配记录到M中
        dis_abs = D
        match = W_w[0]
        for w in W_w:
            # print(w)
            # tmp是公共祖先层数
            tmp = LCA[w['position']][perturbed_node]
            # print(tmp)
            if tmp < dis_abs:
                dis_abs = tmp
                match = w
        MA.append({
            't': item+1,
            'w': match['id']
        })
        W_w.remove(match)

def cal_pro(epsilon):
    global WT
    # 扰动概率
    for i in range(D+1):# 0-10
        wt[i] = math.exp((4-pow(2,i+2))*epsilon)
    for i in range(D):# 0-9
        WT += pow((c-1),i)*(c-2)*wt[i+1]
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            # M[i][j] = round(wt[LCA[i][j]]/WT , 3)
            M[i][j] = wt[LCA[i][j]]/WT

    # 随机游走概率
    tw[0] = WT
    tw[1] = WT-1
    for i in range(D-1):
        tw[i+2] = tw[i+1]-pow((c-1),i)*(c-2)*wt[i+1]

    print(tw)

    for i in range(D):
        print(i)
        pu[i] = tw[i+1]/tw[i]
    # print('每层向上走的概率：',pu)
    # print('结点0扰动概率：',M[0])




# 初始化点
metrixs = [{'x': 44.075, 'y': -48.174}, {'x': 77.797, 'y': 53.385}, {'x': -94.263, 'y': -23.017}, {'x': -53.297, 'y': -57.462}, {'x': 94.109, 'y': -15.844}, {'x': 44.506, 'y': 26.576}, {'x': -94.975, 'y': 21.425}, {'x': -16.807, 'y': 93.151}, {'x': 54.941, 'y': -6.894}, {'x': 48.717, 'y': -62.556}, {'x': 97.181, 'y': -40.41}, {'x': -33.027, 'y': -48.041}, {'x': 10.183, 'y': -44.95}, {'x': -21.037, 'y': 20.742}, {'x': 80.59, 'y': 34.411}, {'x': 47.394, 'y': -23.435}, {'x': -91.673, 'y': -6.454}, {'x': 5.962, 'y': -35.156}, {'x': -12.51, 'y': 28.872}, {'x': -46.362, 'y': -98.04}, {'x': -92.428, 'y': -46.899}, {'x': 28.021, 'y': -7.38}, {'x': -23.196, 'y': 36.076}, {'x': -49.055, 'y': -86.618}, {'x': -73.888, 'y': 9.037}, {'x': 70.506, 'y': 70.594}, {'x': 51.506, 'y': -85.409}, {'x': -43.629, 'y': 50.151}, {'x': -81.334, 'y': -5.776}, {'x': 83.339, 'y': 88.814}, {'x': 84.603, 'y': -26.43}, {'x': -15.736, 'y': -3.053}, {'x': -62.949, 'y': -70.75}, {'x': 6.571, 'y': -15.749}, {'x': 50.749, 'y': -28.665}, {'x': -99.441, 'y': 3.339}, {'x': 31.794, 'y': -64.201}, {'x': -66.774, 'y': 20.075}, {'x': 51.045, 'y': 59.553}, {'x': -49.199, 'y': -80.257}, {'x': 38.845, 'y': 4.306}, {'x': 39.033, 'y': 82.346}, {'x': 81.578, 'y': 30.96}, {'x': 82.775, 'y': 26.634}, {'x': 9.817, 'y': -88.828}, {'x': 40.079, 'y': 47.86}, {'x': 39.38, 'y': -23.268}, {'x': -3.693, 'y': -52.183}, {'x': -95.731, 'y': 6.283}, {'x': -72.161, 'y': -78.876}, {'x': 10.97, 'y': -68.364}, {'x': 84.523, 'y': 61.921}, {'x': 22.847, 'y': -90.604}, {'x': -65.708, 'y': -84.935}, {'x': -17.878, 'y': 30.988}, {'x': 83.912, 'y': 74.289}, {'x': -97.816, 'y': -59.614}, {'x': -30.918, 'y': 79.858}, {'x': 32.66, 'y': 9.239}, {'x': 39.791, 'y': 1.009}, {'x': 74.14, 'y': -99.141}, {'x': 44.575, 'y': 40.96}, {'x': 78.399, 'y': -63.244}, {'x': -24.526, 'y': -32.929}, {'x': 57.895, 'y': 69.837}, {'x': -12.477, 'y': 88.375}, {'x': -28.988, 'y': 8.711}, {'x': 83.147, 'y': 4.556}, {'x': -71.759, 'y': -93.424}, {'x': 26.396, 'y': -82.95}, {'x': 2.836, 'y': -90.622}, {'x': 77.391, 'y': 89.897}, {'x': 70.883, 'y': 81.639}, {'x': 62.948, 'y': -77.977}, {'x': -93.13, 'y': -30.763}, {'x': -37.138, 'y': 81.429}, {'x': 53.956, 'y': -40.426}, {'x': 45.013, 'y': 40.205}, {'x': -40.55, 'y': 90.244}, {'x': -77.144, 'y': 39.439}, {'x': -55.279, 'y': 75.239}, {'x': 78.872, 'y': -11.424}, {'x': 8.896, 'y': 46.878}, {'x': 8.38, 'y': -21.016}, {'x': 61.723, 'y': -11.274}, {'x': 0.059, 'y': 47.11}, {'x': 36.52, 'y': -3.247}, {'x': 91.088, 'y': 93.426}, {'x': 61.384, 'y': -28.893}, {'x': -59.925, 'y': 76.644}, {'x': -93.997, 'y': -33.969}, {'x': 71.025, 'y': -38.832}, {'x': -79.298, 'y': -12.876}, {'x': 14.162, 'y': -67.599}, {'x': 37.25, 'y': -60.656}, {'x': -91.679, 'y': 97.114}, {'x': 93.143, 'y': 89.763}, {'x': -44.423, 'y': -97.143}, {'x': -31.612, 'y': -19.44}, {'x': -91.283, 'y': 9.41}]
# V的一个随机序列PI 
PI = metrixs
random.shuffle(PI) 
print('V的一个随机序列:',PI)
# 树的最高层D   （层数0-1- -D）
maxD = max_dis(metrixs) 
D = math.ceil(math.log(2*maxD,2))
print('level:',D)
num_of_nodes = 0
# beta
beta = random.uniform(0.5,1)
# beta = 0.5
print('beta:',beta)
# 每一层的结点
S = [[] for i in range(D+1)]
# 划分距离
r = [0]*(D)
# maximum number of branches in the tree
c = 0
# epsilon
epsilon = 0.1
WT = 1
wt = [0 for i in range(D+1)]
tw = [0 for i in range(D+1)]
pu = [0 for i in range(D+1)]


######数据集读取#####
lt = 3000
lw = 5000
m = 100
sd = 20

fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
test_data = fo.readlines()
fo.close()
task_test = eval(test_data[0])
worker_test = eval(test_data[1])
print(len(task_test))
print(len(worker_test))
# 测试worker集合和task集合
workers = worker_test
tasks = task_test


######代码运行######

# algorithm_1构造树
HST_tree = algorithm_1(metrixs)
print('分支数：',c)
c = c + 1
# 叶子结点数
num_of_nodes = pow(c-1, D)
# 初始化概率矩阵
M = [[0 for i in range(num_of_nodes)] for i in range(num_of_nodes)]
# LCA[x][a]，任意两个结点的最近公共祖先所在层
LCA = [[0 for i in range(num_of_nodes)] for i in range(num_of_nodes)]
LCA_level(D, 0, num_of_nodes-1)
# 构建S，每一层的结点和对应的父亲结点如下
get_S(HST_tree, D)
# 计算概率矩阵
cal_pro(epsilon)
# 测试
W_w = []
MA = []
# worker
for i in range(len(workers)):
    # 返回扰动的下标
    re = worker_peturbed(workers[i])
    tmp = {
        'id' : i+1,
        'position' : re
    }
    W_w.append(tmp)
# print(W_w)
algorithm_4(tasks)
print('匹配结果：',MA)

# 待完成的内容：
# 根据MA计算总距离