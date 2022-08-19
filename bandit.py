import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

from numpy.lib.shape_base import split
ap=argparse.ArgumentParser()
ap.add_argument('--instance')
ap.add_argument('--algorithm')
ap.add_argument('--randomSeed')
ap.add_argument('--scale')
ap.add_argument('--threshold')
ap.add_argument('--horizon')
ap.add_argument('--epsilon')
arguments= ap.parse_args()
ins = arguments.instance
In=[]
with open (ins) as f:
    content =f.readlines()
try:
    for i in content:

        In.append(float(i))
except:
    for i in content :
        q=i.split()
        r=[]
        for s in q:
            r.append(float(s))
        In.append(r)    

#In is the list of reward probability of an instance
al=arguments.algorithm # al is the algorithm
rs=int(arguments.randomSeed) # rs is the randomseed self argument
c=float(arguments.scale) #scaling
th=float(arguments.threshold) # thershold
hz=int(arguments.horizon)  # horizon
ep=float(arguments.epsilon)
#print(al,rs,c,th,hz)
# python bandit.py --instance assignment1\cs747-pa1-v1\instances\instances-task1\i-1.txt --algorithm epsilon-greedy --randomSeed 123 --scale 1 --threshold 0 --horizon 100
algorithm= ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']

def epsilon_greedy(In,hz,rs,ep):
    np.random.seed(rs)
    history= {}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for j in range(hz):
        if np.random.random()<ep:
            arm_picked=np.random.randint(0,len(In)) # gives the index of the arm 
        else :
            d=0
            arm_picked=0
            for i in range(len(In)):
                if history[i][1]>d:
                    d=history[i][1]
                    arm_picked=i
        # now we have got the arm picked
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history

def ucb(rs,hz,In):

    np.random.seed(rs)
    history={}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for i in range(len(In)):
        if np.random.random() < In[i]:
            history[i][0]+=1
            history[i][2]+=1
            history[i][1]= history[i][2]/history[i][0]
        else :
            history[i][0]+=1
            
            history[i][1]= history[i][2]/history[i][0]
    # using above for loop we pulled each arm once and now we will apply ucb 
    for j in range(len(In),hz):
        arm_picked=0
        ucb=0
        for i in range(len(In)):
            if history[i][1] + np.sqrt((2*np.log(j))/history[i][0])> ucb:
                ucb = history[i][1] + np.sqrt((2*np.log(j))/history[i][0])
                arm_picked = i
        # now we have got which arm to pick
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history
def kl(x,y):
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y)) 



def kl_ucb(rs,hz,In):
    np.random.seed(rs)
    history ={}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for i in range(len(In)):
        if np.random.random() < In[i]:
            history[i][0]+=1
            history[i][2]+=1
            history[i][1]= history[i][2]/history[i][0]
        else :
            history[i][0]+=1
            
            history[i][1]= history[i][2]/history[i][0]
    
    # using above for loop we pulled each arm once and now we will apply ucb-kl
    for j in range(len(In),hz):

        arm_picked=0
        ucb_kl_max=0
        for i in range(len(In)):
            c     = 3
            tol   = 1.0e-4
            p=history[i][1]

            start = p
            end   = 1.0
            mid   = (start + end) / 2.0
            final = (np.log(j) + c*np.log(np.log(j))) / history[i][0]

            while abs(start - end) > tol:
                if p*np.log(p/mid) + (1-p)*np.log((1-p)/(1-mid)) > final:
                    end   = mid
                else:
                    start = mid
                    
                mid = (start + end) / 2.0
            #print(mid)
            if mid > ucb_kl_max:
                ucb_kl_max=mid
                arm_picked=i
                
        
            
        
    # now we have picked the arm
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history

def thompson_sampling(rs,hz,In):
    np.random.seed(rs)
    # lets pull each arm first before applying thompson's sampling
    history ={}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for i in range(len(In)):
        if np.random.random() < In[i]:
            history[i][0]+=1
            history[i][2]+=1
            history[i][1]= history[i][2]/history[i][0]
        else :
            history[i][0]+=1
            
            history[i][1]= history[i][2]/history[i][0]
    for j in range(len(In),hz):
        beta_max=0
        arm_picked=0
        for i in range(len(In)):

            beta= np.random.beta(history[i][2]+1,history[i][0]-history[i][2]+1)
            if beta>beta_max:
                beta_max=beta
                arm_picked=i
        # now we have selected which arm we have to pick using thompson sampling  method
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history
def ucb_t2(rs,hz,In,c):

    np.random.seed(rs)
    history={}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for i in range(len(In)):
        if np.random.random() < In[i]:
            history[i][0]+=1
            history[i][2]+=1
            history[i][1]= history[i][2]/history[i][0]
        else :
            history[i][0]+=1
            
            history[i][1]= history[i][2]/history[i][0]
    # using above for loop we pulled each arm once and now we will apply ucb 
    for j in range(len(In),hz):
        arm_picked=0
        ucb=0
        for i in range(len(In)):
            if history[i][1] + np.sqrt((2*np.log(j))/history[i][0])> ucb:
                ucb = history[i][1] + np.sqrt((c*np.log(j))/history[i][0])
                arm_picked = i
        # now we have got which arm to pick
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history

# task 2 for finding the scaling factor 
    
def ucb_t2(rs,hz,In,c):

    np.random.seed(rs)
    history={}
    for i in range(len(In)):
        history[i] = [0,0,0]  # [no of times picked, emperical mean, total reward]
    for i in range(len(In)):
        if np.random.random() < In[i]:
            history[i][0]+=1
            history[i][2]+=1
            history[i][1]= history[i][2]/history[i][0]
        else :
            history[i][0]+=1
            
            history[i][1]= history[i][2]/history[i][0]
    # using above for loop we pulled each arm once and now we will apply ucb 
    for j in range(len(In),hz):
        arm_picked=0
        ucb=0
        for i in range(len(In)):
            if history[i][1] + np.sqrt((2*np.log(j))/history[i][0])> ucb:
                ucb = history[i][1] + np.sqrt((c*np.log(j))/history[i][0])
                arm_picked = i
        # now we have got which arm to pick
        if np.random.random()<In[arm_picked]:
            history[arm_picked][0]+=1
            history[arm_picked][2]+=1
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
        else:
            history[arm_picked][0]+=1
            
            history[arm_picked][1] = history[arm_picked][2]/history[arm_picked][0]
    return history
    
def al_t3(rs,hz,In):
    np.random.seed(rs)
    reward = In[0]
    In_change=In[1:]
    history={}
    for i in range(len(In_change)):
        history[i]=[0,0,[]]  # [no. of times pulled, reward, list of rewards in sequence]
    # lets pull each arm first
    for i in range(len(In_change)):
        history[i][0]+=1
        r=np.random.choice(reward,p=In_change[i])
        history[i][1]+=r
        history[i][2].append(r)
    #now lets apply thompson's sampling
    for j in range(len(In_change),hz):
        beta_max=0
        arm_picked=0
        for i in range(len(In_change)):

            beta= np.random.beta(history[i][1]+1,history[i][0]-history[i][1]+1)
            if beta>beta_max:
                beta_max=beta
                arm_picked=i
        # now the arm is picked
        history[arm_picked][0]+=1
        r=np.random.choice(reward,p=In_change[arm_picked])
        history[arm_picked][1]+=r
        history[arm_picked][2].append(r)
    return history # [no. of times pulled, reward, list of rewards in sequence]

def al_t4(rs,hz,In,th):
    np.random.seed(rs) 
    reward= In[0]
    In_change=In[1:]
    rp_for_each_arm=np.zeros(len(In_change))
    for i in range(len(In[0])):
        if reward[i]>th:
            for k in range(len(In_change)):
                rp_for_each_arm[k]+=In_change[k][i]
            
    return rp_for_each_arm , thompson_sampling(rs,hz,rp_for_each_arm) # this  gives history[i]=[pulls, empirical mean, success] 

def regret_t4(rs,hz,In,th):
    rp,history= al_t4(rs,hz,In,th)
    sum_reward=0
    for i in range(len(In)-1):
        sum_reward += history[i][2]
    reg= max(rp)*hz - sum_reward 
    return reg

def high(rs,hz,In,th):
    rp,history= al_t4(rs,hz,In,th)
    sum_reward=0
    for i in range(len(In)-1):
        sum_reward += history[i][2]
    return sum_reward



   


def regret_t3(rs,hz,In):
    history= al_t3(rs,hz,In)
    
    max_iter=0
    for j in range(1,len(In)):
        expec=0
        for i in range(len(In[0])):
            expec += In[0][i]*In[j][i]
        max_iter=max(max_iter,expec)
    sum_reward=0
    for i in range(len(In)-1):
        sum_reward += history[i][1]
    reg= max_iter*hz - sum_reward     
    return reg




def regret_t1(In,al,hz,rs,ep):
    if al=='epsilon-greedy-t1':
        history=epsilon_greedy(In,hz,rs,ep)
    elif al=='ucb-t1':
        history=ucb(rs,hz,In)
    elif al == 'kl-ucb-t1':
        history=kl_ucb(rs,hz,In)
    elif al == 'thompson-sampling-t1':
        history= thompson_sampling(rs,hz,In)
    
    
    total_reward=0
    for i in range(len(In)):
        total_reward += history[i][2]
    pstar=max(In)
    reg= pstar*hz-total_reward

    return reg 
def regret_t2(In,al,hz,rs,ep,c):
    if al=='ucb-t2':

        history = ucb_t2(rs,hz,In,c)
    
    total_reward=0
    for i in range(len(In)):
        total_reward += history[i][2]
    pstar=max(In)
    reg= pstar*hz-total_reward

    return reg 





if al[-1]=='1':
    print(str(ins)+', '+str(al)+', '+str(rs)+', '+str(ep)+', '+str(2)+ ', '+str(0)+', '+ str(hz)+', '+str(regret_t1(In,al,hz,rs,ep))+ ', '+'0\n')
if al[-1]=='2':
    print(str(ins)+', '+str(al)+', '+str(rs)+', '+str(.02)+', '+str(c)+ ', '+str(0)+', '+ str(hz)+', '+str(regret_t2(In,al,hz,rs,ep,c))+ ', '+'0\n')

if al[-1]=='3':
    print(str(ins)+', '+str(al)+', '+str(rs)+', '+str(.02)+', '+str(2)+ ', '+str(0)+', '+ str(hz)+', '+str(regret_t3(rs,hz,In))+ ', '+'0\n')
if al[-1]=='4':
    print(str(ins)+', '+str(al)+', '+str(rs)+', '+str(.02)+', '+str(2)+ ', '+str(th)+', '+ str(hz)+', '+str(regret_t4(rs,hz,In,th))+ ', '+str(high(rs,hz,In,th))+ '\n')

        













        

















        



    



                

        
