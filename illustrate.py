from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def _make_rotation_transform(angle_x, angle_y, angle_z):
    # Rotation Mat X
    rx = np.asarray(
        [[1, 0, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x), 0],
         [0, np.sin(angle_x), np.cos(angle_x), 0], [0, 0, 0, 1]])

    # Rotation Mat Y
    ry = np.asarray(
        [[np.cos(angle_y), 0, np.sin(angle_y), 0], [0, 1, 0, 0],
         [-np.sin(angle_y), 0, np.cos(angle_y), 0], [0, 0, 0, 1]])

    # Rotation Mat Z
    rz = np.asarray([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                     [np.sin(angle_z), np.cos(angle_z), 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])
    return rx, ry, rz


def _apply_transform(flatten_grid, trns):
    result_grid = np.zeros((flatten_grid.shape))
    for i in range(result_grid.shape[0]):
        result_grid[i][:] = flatten_grid[i][:].T.dot(trns).T
    return result_grid


def rotate(x, y, z, angle_x, angle_y, angle_z):
    flattengrid = np.stack((x.flatten(), y.flatten(), z.flatten(), np.ones((x.size, ))))
    rx, ry, rz = _make_rotation_transform(angle_x, angle_y, angle_z)
    flattengrid = _apply_transform(flattengrid.T, rx)
    flattengrid = _apply_transform(flattengrid, ry)
    flattengrid = _apply_transform(flattengrid, rz).T
    print('== flatten grid:', flattengrid.shape)

    x = flattengrid[0, :].reshape(x.shape)
    y = flattengrid[1, :].reshape(y.shape)
    z = flattengrid[2, :].reshape(z.shape)
    return x, y, z

# Draw the 2.5D Schema
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(-40, 50, 10), np.arange(-40, 50, 10))
z = np.zeros(x.shape) 
c = [0, 0, 0.5, 0.2]
ax.plot_surface(x, y, z, color=c)
ax.plot([0, 0], [0,0], [40, -40], color='k')

x, y, z = rotate(x, y, z, np.pi/2, 0, 0)
ax.plot_surface(x, y, z, color=c)
ax.plot([0, 0], [40,-40], color='k')
ax.plot([40, -40], [0,0], color='k')

x, y, z = rotate(x, y, z, 0, 0, np.pi / 2)
ax.plot_surface(x, y, z, color=c)
plt.axis('off')
plt.savefig('25d-grid.png', dpi=300)

# Draw TF 2.5D
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(-40, 50, 10), np.arange(-40, 50, 10))
z = np.zeros(x.shape) 
c = [0, 0, 0.5, 0.2]
ax.plot_surface(x, y, z, color=c)
ax.plot([0, 0], [0,0], [40, -40], color='k')

x1, y1, z1 = rotate(x, y, z, 0,  np.pi/4, 0)
x2, y2, z2 = rotate(x, y, z, 0, -np.pi/4, 0)
ax.plot_surface(x1, y1, z1, color=c)
ax.plot_surface(x2, y2, z2, color=c)

x, y, z = rotate(x, y, z, np.pi/2, 0, 0)
ax.plot_surface(x, y, z, color=c)
ax.plot([0, 0], [40,-40], color='k')
ax.plot([40, -40], [0,0], color='k')

x1, y1, z1 = rotate(x, y, z, np.pi/4, 0, 0)
x2, y2, z2 = rotate(x, y, z, -np.pi/4, 0, 0)
ax.plot_surface(x1, y1, z1, color=c)
ax.plot_surface(x2, y2, z2, color=c)

x, y, z = rotate(x, y, z, 0, 0, np.pi / 2)
ax.plot_surface(x, y, z, color=c)

x1, y1, z1 = rotate(x, y, z, 0, 0, np.pi/4)
x2, y2, z2 = rotate(x, y, z, 0, 0, -np.pi/4)
ax.plot_surface(x1, y1, z1, color=c)
ax.plot_surface(x2, y2, z2, color=c)

# Draw TF 2.5D A
f = plt.figure()
ax = f.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(-40, 50, 10), np.arange(-40, 50, 10))
z = np.zeros(x.shape) 
c = [0, 0, 0.5, 0.2]
ax.plot([0, 0], [-40, 40], color='k')
ax.plot_surface(x, y, z, color=c)

x1, y1, z1 = rotate(x, y, z, 0,  np.pi/4, 0)
x2, y2, z2 = rotate(x, y, z, 0, -np.pi/4, 0)
ax.plot_surface(x1, y1, z1, color=c)
ax.plot_surface(x2, y2, z2, color=c)

plt.axis('off')
plt.savefig('25d-grid.png', dpi=300)
plt.show()