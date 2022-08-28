from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10]
# y_1 = [(1.3794+0.0475)/2, (0.05707+0.1528)/2, (0.03767+0.0811)/2, (0.0156, 0.0108, 0.0140, 0.0054, 0.0140, 0.0080, 0.0083]
y_2 = [(0.0875+0.0882)/2, (0.0166+0.0922)/2, (0.0494+0.0264)/2, (0.0205+0.0426)/2, (0.0349+0.0068)/2, (0.0032+0.0350)/2, (0.0133+0.0152)/2, (0.0134+0.0043)/2, (0.0040+0.0034)/2, (0.0006+0.0041)/2]

fig, ax = plt.subplots()
# ax.plot(x, y_1, label='No.1')
ax.plot(x,y_2,label='Loss')
ax.set_xlabel('epoch times')
ax.set_ylabel('loss')
ax.set_title('Draw Loss Curve')
ax.legend()
plt.show()