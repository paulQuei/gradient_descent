import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

max_x = 10
data_size = 10
theta_0 = 5
theta_1 = 2
learning_rate = 0.01

def get_data():
  x = np.linspace(1, max_x, data_size)
  noise = np.random.normal(0, 0.2, len(x))
  y = theta_0 + theta_1 * x + noise
  print(x)
  print(y)
  return x, y

def draw_data(x, y):
  plt.scatter(x,y)

def draw_linear_model(t0, t1, color):
  x = np.arange(0, max_x + 1)
  y = t0 + t1 * x
  plt.plot(x, y, color)

def cost_function(x, y, t0, t1):
  cost_sum = 0
  for i in range(len(x)):
    cost_item = np.power(t0 + t1 * x[i] - y[i], 2)
    cost_sum += cost_item
  return cost_sum / len(x)

def draw_cost(x, y):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.gca(projection='3d')
  scatter_count = 100
  radius = 1
  t0_range = np.linspace(theta_0 - radius, theta_0 + radius, scatter_count)
  t1_range = np.linspace(theta_1 - radius, theta_1 + radius, scatter_count)
  cost = np.zeros((len(t0_range), len(t1_range)))
  for a in range(len(t0_range)):
    for b in range(len(t1_range)):
      cost[a][b] = cost_function(x, y, t0_range[a], t1_range[b])
  t0, t1 = np.meshgrid(t0_range, t1_range)

  ax.set_xlabel('theta_0')
  ax.set_ylabel('theta_1')
  ax.plot_surface(t0, t1, cost, cmap=cm.hsv)

  for angle in range(95, 180, 3):
    ax.set_zlabel('Angle: ' + str(angle))
    ax.view_init(30, angle)
    filename = './' + str(angle) + '.png'
    fig.tight_layout()
    fig.savefig(filename)
    print('Save ' + filename + ' finish')

def gradient_descent(x, y):
  t0 = 10
  t1 = 10
  delta = 0.001
  save_fig = True
  save_time_interval = 50
  save_time = 0
  for times in range(1000):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
      sum1 += (t0 + t1 * x[i] - y[i])
      sum2 += (t0 + t1 * x[i] - y[i]) * x[i]
    t0_ = t0 - 2 * learning_rate * sum1 / len(x)
    t1_ = t1 - 2 * learning_rate * sum2 / len(x)
    print('Times: {}, gradient: [{}, {}]'.format(times, t0_, t1_))
    if save_fig:
      if save_time == 0 or times - save_time > save_time_interval:
        filename = './' + '{:05d}'.format(times) + '.png'
        draw_linear_model(t0_, t1_, 'b')
        plt.savefig(filename)
        save_time = times
    if (abs(t0 - t0_) < delta and abs(t1 - t1_) < delta):
      print('Gradient descent finish')
      return t0_, t1_
    t0 = t0_
    t1 = t1_
  print('Gradient descent too many times')
  return t0, t1

def gradient_descent_with_momentum(x, y):
  t0 = 10
  t1 = 10
  delta = 0.001
  save_fig = True
  save_time_interval = 10
  save_time = 0
  v0 = 0
  v1 = 0
  gamma = 0.9
  for times in range(1000):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
      sum1 += (t0 + t1 * x[i] - y[i])
      sum2 += (t0 + t1 * x[i] - y[i]) * x[i]
    v0 = gamma * v0 + 2 * learning_rate * sum1 / len(x)
    v1 = gamma * v1 + 2 * learning_rate * sum2 / len(x)
    t0_ = t0 - v0
    t1_ = t1 - v1
    print('Times: {}, gradient: [{}, {}]'.format(times, t0_, t1_))
    if save_fig:
      if save_time == 0 or times - save_time > save_time_interval:
        filename = './' + '{:05d}'.format(times) + '.png'
        draw_linear_model(t0_, t1_, 'b')
        plt.savefig(filename)
        save_time = times
    if (abs(t0 - t0_) < delta and abs(t1 - t1_) < delta):
      print('Gradient descent finish')
      return t0_, t1_
    t0 = t0_
    t1 = t1_
  print('Gradient descent too many times')
  return t0, t1

def predict(a, b, x):
  return a * x + b

if __name__ == '__main__':
  np.random.seed(59)
  x, y = get_data()
  plt.ylim(np.min(y) - 1, np.max(y) + 1)
  plt.tight_layout()
  draw_data(x, y)
  # draw_linear_model(10, 0, 'g')
  # draw_linear_model(8, 1, 'b')
  # draw_linear_model(6, 1.5, 'r')
  # draw_cost(x, y)
  gradient_descent_with_momentum(x, y)
  plt.show()