import os
import sys
sys.path.append(os.path.abspath(os.path.join(".")))
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D # import enables 3d plotting
import image_processing.camera_model as camera_model
import math
import numpy

''' creates a simple plot for showing functioning position estimation '''

def projection(impl=0):
    camera_position = [5,1,1]
    #camera_position = [0,0,0]
    rotation_vector = [3,0,0]
    #rotation = [0,0,0]
    em = camera_model.ExtrinsicModel(direction=rotation_vector)
    em_translation = numpy.matmul(em.getRotationMatrix(), numpy.matrix(camera_position).transpose()).transpose().tolist()[0]
    em = camera_model.ExtrinsicModel(translationVector=em_translation, direction=rotation_vector)
    cm = camera_model.CameraModel(em=em)

    real_coords = [20,20,10]

    img_coords = cm.projectToImage(real_coords)
    real_coords_calc = cm.projectToWorld(img_coords, implementation=impl)
    real_coords_calc.length_factor(1)
    closest_point = real_coords_calc.closest_point(real_coords).transpose().tolist()[0]
    print("camera_position: {}\nreal_coord: {}\nclosestp: {}, \nVector: {}\n".format(camera_position, closest_point, real_coords, real_coords_calc.vector))

    x = [real_coords[0], camera_position[0], closest_point[0]]
    y = [real_coords[1], camera_position[1], closest_point[1]]
    z = [real_coords[2], camera_position[2], closest_point[2]]

    print("x: {}\nY: {}\nZ: {}\n".format(x,y,z))

    # translation: blue
    # real coord : cyan
    # closestpoint:yellow
    color = ['b', 'y', 'c']

    plot_in_3d(x, y, z, vectors=[real_coords_calc], colors=color)

def plot_in_3d(x, y, z, vectors=[], colors=['b']):
    fig = pyplot.figure()
    # axis
    plot = fig.add_subplot(111, projection='3d')
    plot.set_xlabel('X-axis')
    plot.set_ylabel('Y-axis')
    plot.set_zlabel('Z-axis')
    # lines to points
    # X-line
    [plot.plot([0, dot_x], [dot_y, dot_y], [dot_z, dot_z], '-', linewidth=2, c='b', alpha=0.3) for dot_x, dot_y, dot_z in
     zip(x, y, z)]
    # Y-line
    [plot.plot([dot_x, dot_x], [0, dot_y], [dot_z, dot_z], '-', linewidth=2, c='m', alpha=0.3) for dot_x, dot_y, dot_z in
     zip(x, y, z)]
    # Z-line
    [plot.plot([dot_x, dot_x], [dot_y, dot_y], [0, dot_z], '-', linewidth=2, c='b', alpha=0.3) for dot_x, dot_y, dot_z in
     zip(x, y, z)]

    for vector in vectors:

        start_point = vector.start_point.transpose().tolist()[0]
        next_point = (vector.start_point + vector.vector*3).transpose().tolist()[0]
        p = list(zip(start_point,next_point))
        plot.plot(p[0], p[1], p[2], '-', linewidth=3, c='r')
        plot.plot([start_point[0]],[start_point[1]],[start_point[2]], marker='o', c='k')
    print("scatter")
    # points
    plot.scatter(x, y, z, marker='o', c=colors, s=100, alpha=1)
    #pyplot.ioff()
    pyplot.show()

def test_plot():
    x = [0, 2, -3, -1.5]
    y = [0, 3, 1, -2.5]
    z = [1, 2, 3, 4]
    plot_in_3d(x,y,z)

if __name__ == "__main__":
    for i in range(1):
        projection(impl=i)

