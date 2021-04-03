from numba import jit
import numpy as np
from numpy import cos, sin, sqrt, dot, pi
from random import random


def culculate_R(axis, angle):
    x, y, z = axis

    R = np.array([
        [cos(angle) + (1 - cos(angle))*x**2, (1 - cos(angle))*x*y - sin(angle)*z, (1 - cos(angle))*x*z + sin(angle)*y],
        [(1 - cos(angle))*y*x + sin(angle)*z, cos(angle) + (1 - cos(angle))*y**2, (1 - cos(angle))*y*z - sin(angle)*x],
        [(1 - cos(angle))*z*x - sin(angle)*y, (1 - cos(angle))*z*y + sin(angle)*x, cos(angle) + (1 - cos(angle)*z**2)]
    ])

    return R

def culculate_R_euler(euler_angles):
    alpha, beta, gamma = euler_angles

    Rx = np.array([
        [1,                 0,              0               ],
        [0,                 cos(beta),   sin(beta)    ],
        [0,                 -sin(beta),  cos(beta)    ]
    ])

    Ry = np.array([
        [cos(gamma),     0,              -sin(gamma)  ],
        [0,                 1,              0               ],
        [sin(gamma),     0,              cos(gamma)   ]
    ])

    Rz = np.array([
        [cos(alpha),     sin(alpha),  0               ],
        [-sin(alpha),    cos(alpha),  0               ],
        [0,                 0,              1               ]
    ])

    Rzx = dot(Rz, Rx)
    R = dot(Rzx, Ry)
    return R


@jit(nopython=True, cache=True)
def rotating_the_coordinates(coordinates, R):
    for i in range(coordinates.shape[0]):
        coordinates[i] = dot(R, coordinates[i])


@jit(nopython=True, cache=True)
def generate_directions(n):
    directions = np.empty((n, 3))
    for i in range(n):
        a1 = random()
        a2 = random()
        cos_alpha = 1 - 2*a1
        sq = sqrt(1 - cos_alpha**2)
        cos_beta = sq*cos(2*pi*a2)
        cos_gamma = sq*sin(2*pi*a2)
        directions[i] = (cos_alpha, cos_beta, cos_gamma)
    return directions


@jit(nopython=True, cache=True)
def withDecayTime(time, decayTime):
    timeWithDecay = np.zeros_like(time)
    countdownTime = 0.
    for i, t in enumerate(time):
        if (t - countdownTime) > decayTime:
            countdownTime = t
        timeWithDecay[i] = countdownTime + decayTime
    return timeWithDecay

