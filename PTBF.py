import random
import math


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
        S[level].append(HST_tree[0])
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

def cal_pro(epsilon):
    global WT
    # 扰动概率
    WT = 1
    for i in range(D+1):
        wt[i] = math.exp((4-pow(2,i+2))*epsilon)
    for i in range(D):
        WT += pow((c-1),i)*(c-2)*wt[i+1]
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            M[i][j] = round(wt[LCA[i][j]]/WT , 3)

    # 随机游走概率
    tw[0] = WT
    tw[1] = WT-1
    for i in range(D-1):
        tw[i+2] = tw[i+1]-pow((c-1),i)*(c-2)*wt[i+1]
    for i in range(D):
        pu[i] = round(tw[i+1]/tw[i] , 3)
    # print('每层向上走的概率：',pu)
    # print('结点0扰动概率：',M[0])


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
    cal_pro(node['epsilon'])
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
        cal_pro(node_list[item]['epsilon'])
        # 调用算法3获得扰动结果
        # print('task',item,'最近的点',shortest_node)
        # 调用算法3获得扰动结果
        perturbed_node = algorithm_3(shortest_node)
        # print('task',item,'扰动结果为',perturbed_node)
        # 分配worker，在树的叶子节点中(也就是W')找到一个最近的点，从W'删除点，将这次匹配记录到M中
        dis_abs = num_of_nodes+1
        match = dict
        for w in W_w:
            # print(w)
            tmp = abs(w['position'] - perturbed_node)
            # print(tmp)
            if tmp < dis_abs:
                dis_abs = tmp
                match = w
        MA.append({
            't': item+1,
            'w': match['id']
        })
        W_w.remove(match)



# 初始化点
metrixs = [
    {'x': 1,'y': 1},
    {'x': 2,'y': 3},
    {'x': 5,'y': 3},
    {'x': 4,'y': 4}
]
# V的一个随机序列PI 
PI = metrixs
# random.shuffle(PI) 
print('V的一个随机序列:',PI)
# 树的最高层D   （层数0-1- -D）
maxD = max_dis(metrixs) 
D = math.ceil(math.log(2*maxD,2))
print('level:',D)
num_of_nodes = 0
# beta
beta = random.uniform(0.5,1)
beta = 0.5
print('beta:',beta)
# 每一层的结点
S = [[] for i in range(5)]
# 划分距离
r = [0]*D
# maximum number of branches in the tree
c = 0
# epsilon
epsilon = 0.1
WT = 1
wt = [0 for i in range(D+1)]
tw = [0 for i in range(D+1)]
pu = [0 for i in range(D+1)]
# 测试数据：worker集合和task集合
workers = [
    {'x': 2,'y': 3,'epsilon': 0.01},
    {'x': 2,'y': 2,'epsilon': 0.1},
    {'x': 5,'y': 5,'epsilon': 0.2},
]

tasks = [
    {'x': 1,'y': 0,'epsilon': 0.1},
    {'x': 1,'y': 3,'epsilon': 0.1},
    {'x': 6,'y': 2,'epsilon': 0.1},
]

######代码运行######
# epsilon = 0.1
# 每层向上走的概率： [0.606, 0.564, 0.304, 0.075, 0]
# 结点0扰动概率： [0.394, 0.264, 0.119, 0.119, 0.024, 0.024, 0.024, 0.024, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# algorithm_1构造树
HST_tree = algorithm_1(metrixs)
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
# cal_pro(epsilon)
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
# print('\n')
algorithm_4(tasks)
print('匹配结果：',MA)