import subprocess
import numpy as  np
import sys
sys.path.append('')
# # task 1
In_1=["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
al_1=['epsilon-greedy-t1' , 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1']
hz_1=[100, 400, 1600, 6400, 25600, 102400]
rs_1a=np.linspace(0,49,50)
rs_1=[]
for m in rs_1a:
    rs_1.append(int(m))
# for i in In_1:
#     for j in al_1:
#         for k in hz_1:
#             for l in rs_1:
#                 subprocess.run('python bandit.py --instance'+' '+i +' --algorithm'+ ' '+ j+' '
#                 +' --randomSeed'+' '+ str(l)+' '+'--threshold 0'+' '+ '--horizon '+' '+str(k)  + ' --scale 2'+ ' --epsilon .02')

# # task 2

c_2 = np.linspace(0.02,0.3,15)
# In_2=["../instances/instances-task2/i-1.txt", "../instances/instances-task2/i-2.txt", "../instances/instances-task2/i-3.txt", "../instances/instances-task2/i-4.txt", "../instances/instances-task2/i-5.txt"]

# for i in In_2:
#     for c in c_2:
#         for r in rs_1:
#             subprocess.run('python bandit.py --instance'+' '+i +' --algorithm ucb-t2'+ ' '+' --randomSeed ' + str(r)+' '+'--threshold 0'+' '+ '--horizon 10000 ' +'--scale ' +str(c) + ' --epsilon .02')

# task 3
In_3=["../instances/instances-task3/i-1.txt", "../instances/instances-task3/i-2.txt"]

for i in In_3:
    for j in hz_1:
        for k in rs_1:
            subprocess.run('python bandit.py --instance'+' '+i +' --algorithm alg-t3'+ ' '
                +' --randomSeed'+' '+ str(k)+' '+'--threshold 0'+' '+ ' --horizon '+' '+str(j) +' --scale 2 ' + ' --epsilon .02')
# task 4
In_4=["../instances/instances-task4/i-1.txt", "../instances/instances-task4/i-2.txt"]
th_4=[0.4,0.6]
for i in In_4:
    for j in hz_1:
        for k in rs_1:
            for l in th_4:
                subprocess.run('python bandit.py --instance'+' '+i +' --algorithm alg-t4'+ ' '
                +' --randomSeed'+' '+ str(k)+' '+'--threshold '+ str(l) +' '+ ' --horizon '+' '+str(j) +' --scale 2' + ' --epsilon .02')

#python bandit.py --instance assignment1\cs747-pa1-v1\instances\instances-task1\i-1.txt 
# --algorithm epsilon-greedy --randomSeed 123 --scale 1 --threshold 0 --horizon 100
#python script.py > output.txt
