import matplotlib.pyplot as plt
import numpy as np

content = None
with open("without_normal.txt", "r") as file:
    content = file.read()

content = content.split("\n")
point_1 = [float(content[i]) for i in range(1,len(content),6)]
point_2 = [float(content[i]) for i in range(2,len(content),6)]
point_3 = [float(content[i]) for i in range(3,len(content),6)]
point_4 = [float(content[i]) for i in range(4,len(content),6)]
k = [float(content[i]) for i in range(0,len(content),6)]

plt.errorbar(k, point_1, yerr=np.array(point_2)/10, marker='o', linestyle='-', capsize=3)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("test data evaluated by training data, without normalization")

plt.show()

plt.errorbar(k, point_3, yerr=np.array(point_4)/10, marker='o', linestyle='-', capsize=3)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("training data evaluated by training data, without normalization")

plt.show()
