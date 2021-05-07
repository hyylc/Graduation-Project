from scipy.special import lambertw
import math
import random


# 默认值
# task数量
lt = 100
# worker数量
lw = 500
# 均值
m = 100
# 标准差
sd = 20
# 隐私预算
epsilon = 0.6

def pre():
    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
    test_data = fo.readlines()
    tasks = eval(test_data[0])
    workers = eval(test_data[1])
    fo.close()
    return tasks,workers

def dis(i,j):
    return round(math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2)),3)

def polar_Laplace(x,y,epsilon):
    # epsilon = 0.1
    # 在极坐标空间扰动
    # print(x,y)
    θ = random.uniform(0,2*math.pi)
    z = random.random()
    # 指定-1分支
    r = (-1.0/epsilon)*(lambertw((z-1)/math.e,k=-1)+1).real
    # print('θ = ',θ*180.0/math.pi,'  r = ',r)
    x = x + 1000*r*math.cos(θ)
    y = y + 1000*r*math.sin(θ)
    # print(x,y)
    return x,y

def Lap_Greedy(workers,tasks,epsilon):
    # worker的位置进行扰动
    W_w = []
    M = []
    for i in range(len(workers)):
        # x,y = polar_Laplace(workers[i]['x'],workers[i]['y'],workers[i]['epsilon'])
        x,y = polar_Laplace(workers[i]['x'],workers[i]['y'],epsilon)
        W_w.append({
            'id' : i,
            'x' : x,
            'y' : y
        })
    # print(W_w)
    # 按用户的顺序进行扰动，分配当前最近的worker
    for i in range(len(tasks)):
        # x,y = polar_Laplace(tasks[i]['x'],tasks[i]['y'],tasks[i]['epsilon'])
        x,y = polar_Laplace(tasks[i]['x'],tasks[i]['y'],epsilon)
        now = {
            'x' : x,
            'y' : y
        }
        match = dict
        min_dis = 1e5
        for j in range(len(W_w)):
            tmp = dis(now,W_w[j]) 
            if tmp < min_dis:
                match = W_w[j]
                min_dis = tmp
        M.append({
            't' : i,
            'w' : match['id']
        })
        W_w.remove(match)

    Lap_Greedy_total_distance = 0
    for i in M:
        Lap_Greedy_total_distance += dis(tasks[i['t']],workers[i['w']])
    # print('Lap_Greedy算法总距离：',Lap_Greedy_total_distance)
    return Lap_Greedy_total_distance


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
#         distance += Lap_Greedy(workers,tasks,epsilon)
#     distance = distance/20
#     print('20次运行结果的均值：Lap_Greedy算法总距离：',distance)
#     T_variety.append(distance)

# print('任务接收方数量取300-700')
# for i in range(len(T_size)):
#     lt = 300
#     lw = W_size[i]
#     m = 100
#     sd = 20
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Lap_Greedy(workers,tasks,epsilon)
#     distance = distance/20
#     print('20次运行结果的均值：Lap_Greedy算法总距离：',distance)
#     W_variety.append(distance)

# print('均值取50-150')
# for i in range(len(T_size)):
#     lt = 300
#     lw = 500
#     m = mean[i]
#     sd = 20
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Lap_Greedy(workers,tasks,epsilon)
#     distance = distance/20
#     print('20次运行结果的均值：Lap_Greedy算法总距离：',distance)
#     m_variety.append(distance)

# print('标准差取10-30')
# for i in range(len(T_size)):
#     lt = 300
#     lw = 500
#     m = 100
#     sd = sigma[i]
#     tasks,workers = pre()
#     distance = 0
#     for times in range(20):
#         distance += Lap_Greedy(workers,tasks,epsilon)
#     distance = distance/20
#     print('20次运行结果的均值：Lap_Greedy算法总距离：',distance)
#     s_variety.append(distance)

# print('隐私预算取五个值')
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
#     epsilon = a
#     for times in range(20):
#         distance += Lap_Greedy(workers,tasks,epsilon)
#     distance = distance/20
#     print('20次运行结果的均值：Lap_Greedy算法总距离：',distance)
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