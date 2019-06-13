import cl_sim_funcs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

'''
mesh1: a cube (face vectors)
mesh2: a rectangular prism (face vectors)
mesh1verts: list of vertices of mesh1
mesh2verts: list of vertices of mesh2
colours1: colours to apply to mesh1 faces
colours2: colours to apply to mesh2 faces
'''

mesh1 = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], #front
                   [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], #right
                   [[1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]], #back
                   [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], #top
                   [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], #left
                   [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]) #bottom

mesh2 = np.array([[[0.5, 0.2, 0], [2, 0.2, 0], [2, 0.2, 0.5], [0.5, 0.2, 0.5]], #front
                   [[2, 0.2, 0], [2, 1.2, 0], [2, 1.2, 0.5], [2, 0.2, 0.5]], #right
                   [[2, 1.2, 0], [2, 1.2, 0.5], [0.5, 1.2, 0.5], [0.5, 1.2, 0]], #back
                   [[2, 0.2, 0.5], [2, 1.2, 0.5], [0.5, 1.2, 0.5], [0.5, 0.2, 0.5]], #top
                   [[0.5, 0.2, 0], [0.5, 1.2, 0], [0.5, 1.2, 0.5], [0.5, 0.2, 0.5]], #left
                   [[0.5, 0.2, 0], [2, 0.2, 0], [2, 1.2, 0], [0.5, 1.2, 0]]]) #bottom

#mesh1verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
#                       [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
#mesh2verts = np.array([[0.5, 0.2, 0], [2, 0.2, 0], [0.5, 1.2, 0], [0.5, 0.2, 0.5],
#                       [0.5, 1.2, 0.5], [2, 0.2, 0.5], [2, 1.2, 0], [2, 1.2, 0.5]])

colours1 = np.array([0, 1, 2, 3, 4, 5])
colours2 = np.array([2, 0, -1, 1, -3, -2])

fig1 = cl_sim_funcs.plot_3d_fields(mesh1, mesh2, colours1, colours2, 'test', cmap=cm.inferno, col_scale=[0, 1])
fig2 = cl_sim_funcs.plot_3d_fields(mesh2, mesh1, colours2, colours1, 'test', cmap=cm.Accent, col_scale=[-1, 1])

