import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile)
    next( plots)
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y


plt.figure()

x, y = readcsv("smooth_rewards.csv")
plt.plot(x, y, 'r', label='G ')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(10,50)  # y轴的最大值
plt.xlim(100, 299600)  # x轴最大值
plt.title('loss', fontsize=16)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Score', fontsize=20)
plt.legend(fontsize=16)
plt.show()
