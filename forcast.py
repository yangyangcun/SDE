'''一、时间'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import time
import random
import matplotlib as mpl






original_data =pd.read_csv('data_month.csv')
data=original_data.iloc[:,1]
data=np.array(data)
data_train=data[:-7]#训练集80%

t =1#时间间隔
dt=t

log_val=[]
for i in range(len(data)-1):
    m=math.log(data[i+1]/data[i])
    log_val.append(m)

x0=data[0]
rt=np.array(log_val).reshape(-1,1)


#生成sigma函数(标准差)
def spliding_sigma(x,size):
    M=[]
    for i in range(x.shape[0]-size+1):
        fang=np.var(x[i:i+size-1])
        m=pow(fang,0.5)
        M.append(m)
    M=np.array(M)
    return M  

splid_sig=spliding_sigma(rt,10)#将滑动窗口的步长设为7
splid_sigma=splid_sig/pow(t,0.5)
#拟合出sigma(t)的函数
T1_s= np.arange(5,(splid_sigma.shape[0]+5))*dt
z_s=np.polyfit(T1_s,splid_sigma,2)
fun_t_sigma=np.poly1d(z_s)
print(fun_t_sigma) #打印拟合的多项式
yvals1=fun_t_sigma(T1_s) #拟合后的u(t)值


def spliding_ut(x,size):
    M=[]
    for i in range(x.shape[0]-size+1):
        n=np.mean(x[i:i+size-1])
        M.append(n)
    M=np.array(M)
    return M  

splid_ut=(spliding_ut(rt,10)+pow(splid_sig,2)/2)/dt#将滑动窗口的步长设为7
splid_ut.shape

#拟合出u(t)的函数（多项式）
T1 = np.arange(5,(splid_ut.shape[0]+5))*dt
z=np.polyfit(T1,splid_ut,3)
fun_t=np.poly1d(z)
print(fun_t) #打印拟合的多项式
yvals=fun_t(T1) #拟合后的u(t)值




e=np.random.randn(data.shape[0]-1+5,10000)

def TRA_MATRIC2(n):#最后返回一个n次轨道（包括预测值）的矩阵
    T=np.arange(data.shape[0]+5)*dt
    ut_esit=fun_t(T)
    sigma_esit=fun_t_sigma(T)
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data.shape[0]-1+5):
        m=traj1[i]+traj1[i]*ut_esit[i]*dt+traj1[i]*sigma_esit[i]*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(x0)
        for i in range(data.shape[0]-1+5):
            m=traj[i]+traj[i]*ut_esit[i]*dt+traj[i]*sigma_esit[i]*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R

def best_tra2(X):#求最优的一条轨道
    M=[]
    X_new=X[:-5,:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data,X_new[:,j])
        M.append(m)  
    min_value = min(M) 
    min_idx = M.index(min_value)
    return min_value,min_idx

data.shape


e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_1=TRA_MATRIC2(10000) 
best_traj2_1=best_tra2(TRA_value2_1)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_2=TRA_MATRIC2(10000) 
best_traj2_2=best_tra2(TRA_value2_2)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_3=TRA_MATRIC2(10000) 
best_traj2_3=best_tra2(TRA_value2_3)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_4=TRA_MATRIC2(10000) 
best_traj2_4=best_tra2(TRA_value2_4)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_5=TRA_MATRIC2(10000) 
best_traj2_5=best_tra2(TRA_value2_5)



test=pd.DataFrame(data=TRA_value2_1)#将数据放进表格
test.to_csv('month_时间_1.csv') #
test=pd.DataFrame(data=TRA_value2_2)#将数据放进表格
test.to_csv('month_时间_2.csv') #
test=pd.DataFrame(data=TRA_value2_3)#将数据放进表格
test.to_csv('month_时间_3.csv') #
test=pd.DataFrame(data=TRA_value2_4)#将数据放进表格
test.to_csv('month_时间_4.csv') #
test=pd.DataFrame(data=TRA_value2_5)#将数据放进表格
test.to_csv('month_时间_5.csv') #



e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_1=TRA_MATRIC2(10000) 
best_traj2_1=best_tra2(TRA_value2_1)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_2=TRA_MATRIC2(10000) 
best_traj2_2=best_tra2(TRA_value2_2)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_3=TRA_MATRIC2(10000) 
best_traj2_3=best_tra2(TRA_value2_3)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_4=TRA_MATRIC2(10000) 
best_traj2_4=best_tra2(TRA_value2_4)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value2_5=TRA_MATRIC2(10000) 
best_traj2_5=best_tra2(TRA_value2_5)



M=(TRA_value2_1[-5:,best_traj2_1[1]]).reshape(-1,1)
M=np.append(M,(TRA_value2_2[-5:,best_traj2_2[1]]).reshape(-1,1),axis=1)
M=np.append(M,(TRA_value2_3[-5:,best_traj2_3[1]]).reshape(-1,1),axis=1)
M=np.append(M,(TRA_value2_4[-5:,best_traj2_4[1]]).reshape(-1,1),axis=1)
M=np.append(M,(TRA_value2_5[-5:,best_traj2_5[1]]).reshape(-1,1),axis=1)            


M_aver= np.average(M, axis=1)


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
dayday=np.arange(1,6,1)
T_sample=np.arange(1,6,1)
ax1.plot(T_sample,M[:,0],c='palegreen',linestyle='--',label='pred',linewidth=3)
ax1.plot(T_sample,M[:,1],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M[:,2],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M[:,3],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M[:,4],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M_aver,c='y',marker='p',label='mean_pred',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(np.arange(1, 6, 1),fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig7.jpeg", dpi=500, bbox_inches='tight')
plt.show()

test=pd.DataFrame(data=M)#将数据放进表格
test.to_csv('month_时间（预测结果）.csv') #

'''二、SBL'''




'''BISDE'''
def FX1(x):
    m=-0.0040*1-329.3200*x+11277.9818*x**(0.5)-23730.1214*np.log(x)+29092.8877*np.exp(1/x)
    return m
def GX1(x):
    m=216.0374*x-7410.2863*x**(0.5)+15614.6170*np.log(x)-19184.9012*np.exp(1/x)
    return m

def TRA_MATRIC4(n):
    traj1=[]#生成第一条轨道
    traj1.append(data[0])
    for i in range(data.shape[0]-1+5):
        m=traj1[i]+FX1(traj1[i])*dt+GX1(traj1[i])*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(data[0])
        for i in range(data.shape[0]-1+5):
            m=traj[i]+FX1(traj[i])*dt+GX1(traj1[i])*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R

def best_tra4(X):#求最优的一条轨道
    M=[]
    X_new=X[:-5,:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data,X_new[:,j])
        M.append(m)  
    min_value = min(M)
    min_idx = M.index(min_value)
    return min_value,min_idx





e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value4_1=TRA_MATRIC4(10000) 
best_traj4_1=best_tra4(TRA_value4_1)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value4_2=TRA_MATRIC4(10000) 
best_traj4_2=best_tra4(TRA_value4_2)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value4_3=TRA_MATRIC4(10000) 
best_traj4_3=best_tra4(TRA_value4_3)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value4_4=TRA_MATRIC4(10000) 
best_traj4_4=best_tra4(TRA_value4_4)

e=np.random.randn(data.shape[0]-1+5,10000)
TRA_value4_5=TRA_MATRIC4(10000) 
best_traj4_5=best_tra4(TRA_value4_5)



test=pd.DataFrame(data=TRA_value2_1)#将数据放进表格
test.to_csv('month_SBL_1.csv') #
test=pd.DataFrame(data=TRA_value2_2)#将数据放进表格
test.to_csv('month_SBL_2.csv') #
test=pd.DataFrame(data=TRA_value2_3)#将数据放进表格
test.to_csv('month_SBL_3.csv') #
test=pd.DataFrame(data=TRA_value2_4)#将数据放进表格
test.to_csv('month_SBL_4.csv') #
test=pd.DataFrame(data=TRA_value2_5)#将数据放进表格
test.to_csv('month_SBL_5.csv') #


M1=(TRA_value4_1[-5:,best_traj4_1[1]]).reshape(-1,1)
M1=np.append(M1,(TRA_value4_2[-5:,best_traj4_2[1]]).reshape(-1,1),axis=1)
M1=np.append(M1,(TRA_value4_3[-5:,best_traj4_3[1]]).reshape(-1,1),axis=1)
M1=np.append(M1,(TRA_value4_4[-5:,best_traj4_4[1]]).reshape(-1,1),axis=1)
M1=np.append(M1,(TRA_value4_5[-5:,best_traj4_5[1]]).reshape(-1,1),axis=1)            




M_aver1= np.average(M1, axis=1)


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
dayday=np.arange(1,6,1)
T_sample=np.arange(1,6,1)
ax1.plot(T_sample,M1[:,0],c='palegreen',linestyle='--',label='pred',linewidth=3)
ax1.plot(T_sample,M1[:,1],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M1[:,2],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M1[:,3],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M1[:,4],c='palegreen',linestyle='--',linewidth=3)
ax1.plot(T_sample,M_aver1,c='y',marker='p',label='mean_pred',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(np.arange(1, 6, 1),fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.1, 0.5)
plt.savefig("Fig12.jpeg", dpi=500, bbox_inches='tight')
plt.show()

test=pd.DataFrame(data=M1)#将数据放进表格
test.to_csv('month_SBL（预测结果）.csv') #


