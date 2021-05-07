# 记录节点数量
num1 = 0
num2 = 0 

fo = open("HST.txt", "r")
HST_info = eval(fo.readlines()[0])
num1 = pow(HST_info['HST_c'],HST_info['HST_D'])
# print(HST_info)
fo.close()


for i in range(25):
    fo = open(str(i)+"_HST.txt", "r")
    HST_info = eval(fo.readlines()[0])
    num2 += pow(HST_info['HST_c'],HST_info['HST_D'])
    # print(HST_info)
    fo.close()


print(num1,num2)