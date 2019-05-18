import numpy as np
import matplotlib.pyplot as plt

def draw_model(a, b):
  x = np.arange(0, 100)
  y = a * x + b
  plt.plot(x, y, 'r')
  plt.show()

if __name__ == '__main__':
  a = 2
  b = 5
  draw_model(a, b)