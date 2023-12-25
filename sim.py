import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# paraemters
tau_E = 0.1
tau_I = 0.002
W_EE = 1.5
R = 0.5
alpha = 0.8

# gain function
def gain(x):
    Theta = -np.tanh(-1) # Theta = 0.76
    Theta = 0.76
    A_max = 2 / (W_EE / np.cosh(-Theta)**2) # A_max = 2.26
    A_max = 2.26
    return 0.5 * A_max * (1 + np.tanh(x - Theta))
def i1(t):
    return 1
def i2(t):
    return 1

# differential equations
def dhdt(t, h, I1, I2):
    h1, h2 = h
    return np.array([
        (-h1 + R * I1 + (W_EE-alpha) * gain(h1) - alpha * gain(h2)) / tau_E,
        (-h2 + R * I2 + (W_EE-alpha) * gain(h2) - alpha * gain(h1)) / tau_E,
    ])

h1_0 = -1
h2_0 = 0
h_0 = (h1_0, h2_0)

h1_1 = -2.3
h2_1 = -0.7
h_1 = (h1_1, h2_1)

# solve differential equations
t = np.linspace(0, 3, 10000)
I = (i1(t), i2(t))
sol = odeint(dhdt, y0=h_0, t=t, tfirst=True, args=I)

h1_sol = sol.T[0]
h2_sol = sol.T[1]
print(h1_sol)
print(h2_sol)

sol1 = odeint(dhdt, y0=h_1, t=t, tfirst=True, args=I)
h1_sol1 = sol1.T[0]
h2_sol1 = sol1.T[1]

# vector field
x0 = np.linspace(-5, 5, 20)
x1 = np.linspace(-5, 5, 20)
# create a grid
X0, X1 = np.meshgrid(x0, x1)
# projections
dX0 = np.zeros(X0.shape)
dX1 = np.zeros(X1.shape)
shape1, shape2 = X1.shape

for index_shape1 in range(shape1):
    for index_shape2 in range(shape2):
        dxdtAtX = dhdt(t, [X0[index_shape1, index_shape2], X1[index_shape1, index_shape2]], *I)
        dX0[index_shape1, index_shape2] = dxdtAtX[0]
        dX1[index_shape1, index_shape2] = dxdtAtX[1]

# plot results
fig1 = plt.figure()
plt.quiver(X0, X1, dX0, dX1, color='k')
plt.contour(X0, X1, dX0, levels=[0], colors='red', linestyles='dashed')
plt.contour(X0, X1, dX1, levels=[0], colors='blue', linestyles='dashed')
plt.plot(h1_0, h2_0, 'o', color='green')
# plt.plot(h1_1, h2_1, 'o', color='orange')
plt.plot(h1_sol, h2_sol, label='Correct', color='green')
# plt.plot(h1_sol1, h2_sol1, label='Wrong', color='orange')
plt.xlabel(r'$h_{E, 1}$', fontsize=12)
plt.ylabel(r'$h_{E, 2}$', fontsize=12)

fig2 = plt.figure()
t = np.linspace(0, 1, 10000)
h_0 = (-0.5, -0.5)

I = (2, 1)
sol = odeint(dhdt, y0=h_0, t=t, tfirst=True, args=I)
h1_sol = sol.T[0]
h2_sol = sol.T[1]

I = (1.5, 1)
sol1 = odeint(dhdt, y0=h_0, t=t, tfirst=True, args=I)
h1_sol1 = sol1.T[0]
h2_sol1 = sol1.T[1]

plt.plot(t, gain(h1_sol), color='b', label='33.3')
plt.plot(t, gain(h2_sol), color='b', ls='--')
plt.plot(t, gain(h1_sol1), color='r', label='20')
plt.plot(t, gain(h2_sol1), color='r', ls='--')
plt.xlabel(r'Time (s)', fontsize=12)
plt.ylabel(r'Activity (Hz)', fontsize=12)
plt.legend(title='Coherence')
plt.show()
