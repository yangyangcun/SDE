import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import time
import random
import matplotlib as mpl   




original_data =pd.read_csv('data_two.csv')
data=original_data.iloc[:,1]
data=np.array(data)
data_train=data[:-7]#训练集80%
data_test=data[-7:]#测试集


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.plot(range(original_data.shape[0]),original_data.iloc[:,1],'lightskyblue',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig1.jpeg", dpi=500, bbox_inches='tight')
plt.show()



e=np.random.randn(data.shape[0]-1,10000)


'''正常参数估计'''
#估计参数：均值和方差
log_val=[]
for i in range(len(data_train)-1):    
    m=math.log(data_train[i+1]/data_train[i])
    log_val.append(m)
u=np.mean(log_val)#样本取对数后的均值
d2=np.var(log_val)#样本取对数后的方差
t =1#时间间隔
dt=t
mu=(u+(d2)/2)/t#随机过程均值
sigma=pow(d2,0.5)/pow(t,0.5)#随机过程标准差
x0=data[0]#趋势项初始值


def TRA_MATRIC1(n):#最后返回一个n次轨道（包括预测值）的矩阵
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data.shape[0]-1):
        m=traj1[i]+traj1[i]*mu*dt+traj1[i]*sigma*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(x0)
        for i in range(data.shape[0]-1):
            m=traj[i]+traj[i]*mu*dt+traj[i]*sigma*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R
def best_tra1(X):#求最优的一条轨道
    M=[]
    X_new=X[:-data_test.shape[0],:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data_train,X_new[:,j])
        M.append(m)  
    min_value = min(M) 
    min_idx = M.index(min_value)
    return min_value,min_idx

TRA_value1=TRA_MATRIC1(10000) 
best_traj1=best_tra1(TRA_value1)

test=pd.DataFrame(data=TRA_value1)#将数据放进表格
test.to_csv('2years_常数.csv') #

paths=TRA_value1[:-data_test.shape[0],:].T
paths.shape
mean_path = np.mean(paths, axis=0)
std_path = np.std(paths, axis=0)

plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_train.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value1[:-data_test.shape[0],909],c='limegreen',label='Realizations',linewidth=3)
ax1.plot(T1,TRA_value1[:-data_test.shape[0],99],c='limegreen',linewidth=3)
ax1.plot(T1, mean_path, label='Mean Path',c='tomato',linestyle='--',linewidth=3)
ax1.fill_between(T1, mean_path - 2 * std_path, mean_path + 2 * std_path, alpha=0.2, label='CL95%')
ax1.plot(T1,data_train,c='lightskyblue',label='Data',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':15})
legend=plt.legend(loc='upper left')
plt.savefig("Fig2.jpeg", dpi=500, bbox_inches='tight')
plt.show()


'''确定项'''
def TRA_MATRIC1_CERTAIN():#最后返回一个n次轨道（包括预测值）的矩阵
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data_train.shape[0]-1):
        m=traj1[i]+traj1[i]*mu*dt
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    return R
TRA_value1_cert=TRA_MATRIC1_CERTAIN()



'''效果表现'''
plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_train.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value1[:-data_test.shape[0],best_traj1[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3)
ax1.plot(T1,TRA_value1_cert,c='y',label='ODE_pred',linestyle='--',linewidth=3)
ax1.plot(T1,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
legend=plt.legend(loc='upper right')   
plt.savefig("Fig3.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()



plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(data_train,TRA_value1[:-data_test.shape[0],best_traj1[1]],c='palegreen',label='Pred')
ax1.plot(data_train,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("True value", fontsize=22)
plt.ylabel("Predicted value", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig3.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_test.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value1[-data_test.shape[0]:,best_traj1[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3,marker='s')
ax1.plot(T1,data_test,c='lightskyblue',label='Real',linewidth=3,marker='p')
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
plt.savefig("Fig3.3.jpeg", dpi=500, bbox_inches='tight')
plt.show()



'''验证集表现'''
pre_train=TRA_value1[:-data_test.shape[0],best_traj1[1]]
pre_test=TRA_value1[-data_test.shape[0]:,best_traj1[1]]


'''评价指标'''
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# MAE
metrics.mean_absolute_error(data_train,pre_train)
metrics.mean_absolute_error(data_test,pre_test)
np.sqrt(metrics.mean_squared_error(data_train,pre_train))
np.sqrt(metrics.mean_squared_error(data_test,pre_test))
mape(data_train,pre_train)
mape(data_test,pre_test)
smape(data_train,pre_train)
smape(data_test,pre_test)




'''改为mu_t'''


#估计参数：方差
log_val=[]
for i in range(len(data_train)-1):
    m=math.log(data_train[i+1]/data_train[i])
    log_val.append(m)
rt=np.array(log_val).reshape(-1,1)
rt.shape

plt.figure(figsize=(10,6),dpi=250)
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(np.arange(rt.shape[0]),rt,c='lightskyblue')
plt.xlabel('DAY')
plt.ylabel('s(t)')
plt.legend()



#生成sigma函数(标准差)
def spliding_sigma(x,size):
    M=[]
    for i in range(x.shape[0]-size+1):
        fang=np.var(x[i:i+size-1])
        m=pow(fang,0.5)
        M.append(m)
    M=np.array(M)
    return M  

splid_sig=spliding_sigma(rt,40)#将滑动窗口的步长设为7
splid_sigma=splid_sig/pow(t,0.5)
#拟合出sigma(t)的函数
T1_s= (np.arange(20,(splid_sigma.shape[0]+20)))*dt
T1_s.shape


z_s=np.polyfit(T1_s,splid_sigma,2)
fun_t_sigma=np.poly1d(z_s)
print(fun_t_sigma) #打印拟合的多项式
yvals1=fun_t_sigma(T1_s) #拟合后的u(t)值


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
T1_s=np.arange(20,(splid_sigma.shape[0]+20))
ax1.plot(T1_s,splid_sigma,c='lightskyblue',linewidth=3)
ax1.plot(T1_s,yvals1,c='palegreen',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("sigma(t)", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.12, 0.5)
plt.savefig("Fig4.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()



def spliding_ut(x,size):
    M=[]
    for i in range(x.shape[0]-size+1):
        n=np.mean(x[i:i+size-1])
        M.append(n)
    M=np.array(M)
    return M  

splid_ut=(spliding_ut(rt,40)+pow(splid_sig,2)/2)/dt#将滑动窗口的步长设为7
splid_ut.shape

T1 = np.arange(20,(splid_ut.shape[0]+20))*dt
z=np.polyfit(T1,splid_ut,2)
fun_t=np.poly1d(z)
print(fun_t)
yvals=fun_t(T1) 


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
T1=np.arange(20,(splid_sigma.shape[0]+20))
ax1.plot(T1,splid_ut,c='lightskyblue',linewidth=3)
ax1.plot(T1,yvals,c='palegreen',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("u(t)", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.12, 0.5)
plt.savefig("Fig4.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()




def TRA_MATRIC2(n):#最后返回一个n次轨道（包括预测值）的矩阵
    T=np.arange(data.shape[0])*dt
    ut_esit=fun_t(T)
    sigma_esit=fun_t_sigma(T)
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data.shape[0]-1):
        m=traj1[i]+traj1[i]*ut_esit[i]*dt+traj1[i]*sigma_esit[i]*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(x0)
        for i in range(data.shape[0]-1):
            m=traj[i]+traj[i]*ut_esit[i]*dt+traj[i]*sigma_esit[i]*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R

def best_tra2(X):#求最优的一条轨道
    M=[]
    X_new=X[:-data_test.shape[0],:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data_train,X_new[:,j])
        M.append(m)  
    min_value = min(M) 
    min_idx = M.index(min_value)
    return min_value,min_idx


TRA_value_spliding=TRA_MATRIC2(10000) 
best_traj_spliding=best_tra2(TRA_value_spliding)

test=pd.DataFrame(data=TRA_value_spliding)#将数据放进表格
test.to_csv('2years_时间.csv') #

'''确定项'''
def TRA_MATRIC2_CERTAIN():#最后返回一个n次轨道（包括预测值）的矩阵
    T=np.arange(data_train.shape[0])*dt
    ut_esit=fun_t(T)
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data_train.shape[0]-1):
        m=traj1[i]+traj1[i]*ut_esit[i]*dt
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    return R

TRA_value_spliding_cert=TRA_MATRIC2_CERTAIN()


'''效果表现'''
plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_train.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value_spliding[:-data_test.shape[0],best_traj_spliding[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3)
ax1.plot(T1,TRA_value_spliding_cert,c='y',label='ODE_pred',linestyle='--',linewidth=3)
ax1.plot(T1,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
legend=plt.legend(loc='upper right')
plt.savefig("Fig5.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(data_train,TRA_value_spliding[:-data_test.shape[0],best_traj_spliding[1]],c='palegreen',label='Pred')
ax1.plot(data_train,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("True value", fontsize=22)
plt.ylabel("Predicted value", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig5.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_test.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value_spliding[-data_test.shape[0]:,best_traj_spliding[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3,marker='s')
ax1.plot(T1,data_test,c='lightskyblue',label='Real',linewidth=3,marker='p')
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
plt.savefig("Fig5.3.jpeg", dpi=500, bbox_inches='tight')
plt.show()


'''测试集表现'''
pre_train2=TRA_value_spliding[:-data_test.shape[0],best_traj_spliding[1]]
pre_test2=TRA_value_spliding[-data_test.shape[0]:,best_traj_spliding[1]]


'''评价指标'''
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

metrics.mean_absolute_error(data_train,pre_train2)
metrics.mean_absolute_error(data_test,pre_test2)
np.sqrt(metrics.mean_squared_error(data_train,pre_train2))
np.sqrt(metrics.mean_squared_error(data_test,pre_test2))
mape(data_train,pre_train2)
mape(data_test,pre_test2)
smape(data_train,pre_train2)
smape(data_test,pre_test2)




'''空间滑动'''
#直接多项式拟合
fx_value=[]
for i in range(data_train.shape[0]-1):
    m=(data_train[i+1]-data_train[i])/dt
    fx_value.append(m)
fx_value=np.array(fx_value)
hanshu_x=data_train[:-1]
#拟合出f(t)的函数
fx_value.shape
z=np.polyfit(hanshu_x,fx_value,3)
FX=np.poly1d(z)
print(FX) #打印拟合的多项式
hanshu_xzhou=np.arange(min(hanshu_x),max(hanshu_x)+1)
yvals=FX(hanshu_xzhou) #拟合后的u(t)值

plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(hanshu_x,fx_value,c='lightskyblue',label='original values',s=40)
ax1.plot(hanshu_xzhou,yvals,'palegreen',label='polyfit values',linewidth=3)
plt.xlabel("x", fontsize=26)
plt.ylabel("f(x)", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig8.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()



f_value=FX(hanshu_x)

gx_value=[]
for i in range(data_train.shape[0]-1):
    m=(data_train[i+1]-data_train[i]-f_value[i])**2/dt
    gx_value.append(m)
gx_value=np.array(gx_value)



z1=np.polyfit(hanshu_x,gx_value,3)
GX=np.poly1d(z1)
print(GX) #打印拟合的多项式
yvals1=GX(hanshu_x) #拟合后的u(t)值

plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(hanshu_x,gx_value,c='lightskyblue',label='original values',s=40)
ax1.plot(hanshu_xzhou,GX(hanshu_xzhou),'palegreen',label='polyfit values',linewidth=3)
plt.xlabel("x", fontsize=26)
plt.ylabel("g(x)", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig8.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()



def TRA_MATRIC3(n):
    traj1=[]#生成第一条轨道
    traj1.append(data[0])
    for i in range(data.shape[0]-1):
        m=traj1[i]+FX(traj1[i])*dt+GX(traj1[i])*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(data[0])
        for i in range(data.shape[0]-1):
            m=traj[i]+FX(traj[i])*dt+GX(traj1[i])*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R

def best_tra3(X):#求最优的一条轨道
    M=[]
    X_new=X[:-data_test.shape[0],:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data_train,X_new[:,j])
        M.append(m)  
    min_value = min(M)
    min_idx = M.index(min_value)
    return min_value,min_idx


# 获取开始时间


TRA_value3=TRA_MATRIC3(10000) 


has_nan = np.isnan(TRA_value3).any()
has_inf = np.isinf(TRA_value3).any()

print("数组中是否包含NaN值：", has_nan)
print("数组中是否包含无穷大值：", has_inf)

inf_columns = np.isnan(TRA_value3).any(axis=0)
arr_without_inf = np.delete(TRA_value3, np.where(inf_columns), axis=1)
TRA_value3=arr_without_inf
TRA_value3.shape
best_traj3=best_tra3(TRA_value3)


test=pd.DataFrame(data=TRA_value3)#将数据放进表格
test.to_csv('2years_多项式.csv') #


'''确定项'''
def TRA_MATRIC3_CERTAIN():
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data_train.shape[0]-1):
        m=traj1[i]+FX(traj1[i])*dt
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    return R
TRA_value3_cert=TRA_MATRIC3_CERTAIN()




'''效果表现'''
plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_train.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value3[:-data_test.shape[0],best_traj3[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3)
ax1.plot(T1,TRA_value3_cert,c='y',label='ODE_pred',linestyle='--',linewidth=3)
ax1.plot(T1,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
legend=plt.legend(loc='upper right')
plt.savefig("Fig9.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()



plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(data_train,TRA_value3[:-data_test.shape[0],best_traj3[1]],c='palegreen',label='Pred')
ax1.plot(data_train,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("True value", fontsize=22)
plt.ylabel("Predicted value", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig9.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_test.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value3[-data_test.shape[0]:,best_traj3[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3,marker='s'
)
ax1.plot(T1,data_test,c='lightskyblue',label='Real',linewidth=3,marker='p')
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
plt.savefig("Fig9.3.jpeg", dpi=500, bbox_inches='tight')
plt.show()




'''测试集表现'''
pre_train3=TRA_value3[:-data_test.shape[0],best_traj3[1]]
pre_test3=TRA_value3[-data_test.shape[0]:,best_traj3[1]]



'''评价指标'''
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# MAE
metrics.mean_absolute_error(data_train,pre_train3)
metrics.mean_absolute_error(data_test,pre_test3)
np.sqrt(metrics.mean_squared_error(data_train,pre_train3))
np.sqrt(metrics.mean_squared_error(data_test,pre_test3))
mape(data_train,pre_train3)
mape(data_test,pre_test3)
smape(data_train,pre_train3)
smape(data_test,pre_test3)




'''BISDE'''

def FX1(x):
    m=-0.0502*x+0.5322*x**(0.5)-0.0172*x**(-0.5)-0.0207*1/x-8.6102*np.log(x)/x
    return m
def GX1(x):
    m=-0.789143*x+22.483278*x**(0.5)-31.002143*np.log(x)
    return m


def TRA_MATRIC4(n):
    traj1=[]#生成第一条轨道
    traj1.append(data[0])
    for i in range(data.shape[0]-1):
        m=traj1[i]+FX1(traj1[i])*dt+GX1(traj1[i])*pow(dt,0.5)*e[i,0]
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    for j in range(n-1):
        traj=[]
        traj.append(data[0])
        for i in range(data.shape[0]-1):
            m=traj[i]+FX1(traj[i])*dt+GX1(traj1[i])*pow(dt,0.5)*e[i,j+1]
            traj.append(m)
        traj=np.array(traj).reshape(-1,1)
        R=np.append(R,traj,axis=1)
    return R

def best_tra4(X):#求最优的一条轨道
    M=[]
    X_new=X[:-data_test.shape[0],:]
    for j in range(X_new.shape[1]):
        m=mean_squared_error(data_train,X_new[:,j])
        M.append(m)  
    min_value = min(M)
    min_idx = M.index(min_value)
    return min_value,min_idx

TRA_value4=TRA_MATRIC4(10000) 
has_nan = np.isnan(TRA_value4).any()
has_inf = np.isinf(TRA_value4).any()
print("数组中是否包含NaN值：", has_nan)
print("数组中是否包含无穷大值：", has_inf)

inf_columns = np.isnan(TRA_value4).any(axis=0)
arr_without_inf = np.delete(TRA_value4, np.where(inf_columns), axis=1)
TRA_value4=arr_without_inf
TRA_value4.shape
best_traj4=best_tra4(TRA_value4)

test=pd.DataFrame(data=TRA_value4)#将数据放进表格
test.to_csv('2years_SBL.csv') #



'''确定项'''
def TRA_MATRIC4_CERTAIN():
    traj1=[]#生成第一条轨道
    traj1.append(x0)
    for i in range(data_train.shape[0]-1):
        m=traj1[i]+FX1(traj1[i])*dt
        traj1.append(m)
    R=np.array(traj1).reshape(-1,1)
    return R
TRA_value4_cert=TRA_MATRIC4_CERTAIN()



'''效果表现'''
plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_train.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value4[:-data_test.shape[0],best_traj4[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3)
ax1.plot(T1,TRA_value4_cert,c='y',label='ODE_pred',linestyle='--',linewidth=3)
ax1.plot(T1,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
legend=plt.legend(loc='upper right')
plt.savefig("Fig10.1.jpeg", dpi=500, bbox_inches='tight')
plt.show()



plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.scatter(data_train,TRA_value4[:-data_test.shape[0],best_traj4[1]],c='palegreen',label='Pred')
ax1.plot(data_train,data_train,c='lightskyblue',label='Real',linewidth=3)
plt.xlabel("True value", fontsize=22)
plt.ylabel("Predicted value", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.savefig("Fig10.2.jpeg", dpi=500, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,6),dpi=300)
mpl.rcParams['font.family'] = 'Times New Roman'
T1 = np.arange(data_test.shape[0])
fig,ax1 = plt.subplots(figsize=(10,6))
ax1.plot(T1,TRA_value4[-data_test.shape[0]:,best_traj4[1]],c='palegreen',linestyle='--',label='SDE_pred',linewidth=3,marker='s')
ax1.plot(T1,data_test,c='lightskyblue',label='Real',linewidth=3,marker='p')
plt.xlabel("Day", fontsize=22)
plt.ylabel("WTI Spot Price (Dollars per Barrel)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=22)
plt.gca().xaxis.set_label_coords(0.5, -0.08)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.rcParams.update({'font.size':20})
plt.savefig("Fig10.3.jpeg", dpi=500, bbox_inches='tight')
plt.show()


'''测试集表现'''
pre_train4=TRA_value4[:-data_test.shape[0],best_traj4[1]]
pre_test4=TRA_value4[-data_test.shape[0]:,best_traj4[1]]


'''评价指标'''
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# MAE
metrics.mean_absolute_error(data_train,pre_train4)
metrics.mean_absolute_error(data_test,pre_test4)
np.sqrt(metrics.mean_squared_error(data_train,pre_train4))
np.sqrt(metrics.mean_squared_error(data_test,pre_test4))
mape(data_train,pre_train4)
mape(data_test,pre_test4)
smape(data_train,pre_train4)
smape(data_test,pre_test4)
  





