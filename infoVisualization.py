import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#train


f=open('train/ISIC-2017_Training_Part3_GroundTruth.csv', 'r')
f.readline()

mel=0
nothing=0
kera=0

aux=[]

for line in f.readlines():
    file=line.split(",")[0]
    melanoma=line.split(",")[1]
    keratosis=line.split(",")[2]

    if float(melanoma)==1:
        mel+=1
    elif float(keratosis)==1:
        kera+=1
    else:
        nothing+=1


f.close()

fig, ax = plt.subplots()
data = np.random.rand(1000)

N, bins, patches = ax.hist([1,6,11],bins=[0,5,10,15], weights=[mel*4,kera*4,nothing], rwidth=0.9)

ax.set_ylabel('Number of occurrences')
ax.get_xaxis().set_visible(False)

custom_lines = [Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="y", lw=4)]
ax.legend(custom_lines, ['Melanoma Occurrences', 'Keratosis Occurrences', 'Nevus Occurrences'])


for i in range(0,1):    
    patches[i].set_facecolor('r')
for i in range(1,2):
    patches[i].set_facecolor('b')
for i in range(2,3):
    patches[i].set_facecolor('y')

plt.show()
