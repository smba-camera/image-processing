import os
import sys
sys.path.append(os.path.abspath(os.path.join(".")))
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D # import enables 3d plotting
import image_processing.camera_model as camera_model
import math
import numpy

def projection():

    camera_position = [10,10,10]
    rotation = [math.pi/2, math.pi, math.pi/4]
    em = camera_model.ExtrinsicModel(rotation=rotation)
    em_translation = numpy.matmul(em.getRotationMatrix(), numpy.matrix(camera_position).transpose()).transpose().tolist()[0]
    em = camera_model.ExtrinsicModel(translationVector=em_translation, rotation=rotation)
    cm = camera_model.CameraModel(em=em)

    real_coords = [20,20,20]

    img_coords = cm.projectToImage(real_coords)
    real_coords_calc = cm.projectToWorld(img_coords)
    closest_point = real_coords_calc.closest_point(real_coords).transpose().tolist()[0]
    print("closestp: {}, real_coord: {}".format(closest_point, real_coords))

    x = [real_coords[0], camera_position[0], closest_point[0]]
    y = [real_coords[1], camera_position[1], closest_point[1]]
    z = [real_coords[2], camera_position[2], closest_point[2]]

    # translation: blue
    # real coord : cyan
    # closestpoint:yellow
    color = ['b', 'y', 'y']

    plot_in_3d(x, y, z, real_coords_calc, colors=color)

def plot_in_3d(x, y, z, vector, colors=['b']):
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

    if vector:
        vector_length_factor = 1
        start_point = vector.start_point.transpose().tolist()[0]
        next_point = (vector.start_point + (vector.vector * vector_length_factor)).transpose().tolist()[0]
        p = list(zip(start_point,next_point))
        plot.plot(p[0], p[1], p[2], '-', linewidth=3, c='r')
        plot.plot([start_point[0]],[start_point[1]],[start_point[2]], marker='o', c='k')

    # points
    plot.scatter(x, y, z, marker='o', c=colors, s=100, alpha=1)
    fig.show()
    input()

def test_plot():
    x = [0, 2, -3, -1.5]
    y = [0, 3, 1, -2.5]
    z = [1, 2, 3, 4]
    plot_in_3d(x,y,z)

if __name__ == "__main__":
    projection()

