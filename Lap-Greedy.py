from scipy.special import lambertw
import math
import random


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



# 默认值
# task数量
lt = 100
# worker数量
lw = 500
# 方差
m = 100
# 方差
sd = 20
fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "r")
test_data = fo.readlines()
tasks = eval(test_data[0])
workers = eval(test_data[1])
fo.close()
M = []
epsilon = 0.6
Lap_Greedy(workers,tasks,epsilon)
# print(M)
print("Lap_Greedy匹配结果大小：",len(M))
# print("PTBF匹配结果：",PTBF_MA)
Lap_Greedy_total_distance = 0
for i in M:
    Lap_Greedy_total_distance += dis(tasks[i['t']],workers[i['w']])
print('Lap_Greedy算法总距离：',Lap_Greedy_total_distance)