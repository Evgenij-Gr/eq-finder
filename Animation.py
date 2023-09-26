"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import os

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    # del_ = state[2] - state[0]
    # den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = gamma_ - lambda_*state[1] - sin(state[0]) + k*sin(state[2]-state[0])

    dydx[2] = state[3]

    # den2 = (L2/L1)*den1
    dydx[3] = gamma_ - lambda_*state[3] - sin(state[2]) + k*sin(state[0]-state[2])

    return dydx


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 100, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
fi1 = 35.0
v1 = 0.0
fi2 = 35.0
v2 = 0.0

gamma_ = 0.8
lambda_ = 0.
k = 0.
# initial state
state = np.radians([fi1, v1, fi2, v2])
# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = sin(y[:, 0])
y1 = -cos(y[:, 0])

x2 = sin(y[:, 2])+0.5
y2 = -cos(y[:, 2])


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line_1, = ax.plot([], [], 'o-', lw=2)
line_2, = ax.plot([], [], 'o-', lw=2)
line_3, = ax.plot([], [], 'k-.')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line_1.set_data([], [])
    line_2.set_data([], [])
    line_3.set_data([], [])
    time_text.set_text('')
    return line_1, line_2, line_3, time_text


def animate(i):
    thisx_1 = [0, x1[i]]
    thisy_1 = [0, y1[i]]
    thisx_2 = [0.5, x2[i]]
    thisy_2 = [0, y2[i]]
    thisx_contact = [x1[i], x2[i]]
    thisy_contact = [y1[i], y2[i]]
    line_1.set_data(thisx_1, thisy_1)
    line_2.set_data(thisx_2, thisy_2)
    line_3.set_data(thisx_contact, thisy_contact)
    time_text.set_text(time_template % (i*dt))
    return line_1, line_2, line_3, time_text


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# pathToDir = 'C:/Users/User/eq-finder/output_files/TwoPendulums/Симуляция'
# imageName = f'{gamma_ =}, {lambda_ =}, {k =}'
# fullOutputName = os.path.join(pathToDir, imageName + '.gif')
# ani.save('animation.gif', writer='Pillow', fps=30)
plt.show()
