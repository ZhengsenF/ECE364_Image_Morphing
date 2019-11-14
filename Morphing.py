#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 13
# #######################################################
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sys
from math import ceil, floor


# This function takes in the full file paths of the text files containing the (x, y)
# coordinates of a list of points, for both the left and right images.
# returns the tuple (leftTriangles, rightTriangles), where each is a list of instances of the Triangle class.
def loadTriangles(leftPointFilePath, rightPointFilePath):
    leftPoints = pointsFromFile(leftPointFilePath)
    rightPoints = pointsFromFile(rightPointFilePath)
    leftDelaunay = Delaunay(leftPoints)
    rightDelaunay = Delaunay(rightPoints)
    leftTriangles = triangleFromDelaunay(leftDelaunay, leftPoints)
    rightTriangles = triangleFromDelaunay(rightDelaunay, rightPoints)
    return tuple([leftTriangles, rightTriangles])


# Takes points that used to generate Delaunay and generated Delaunay
# generate plot
def showDelaunay(points, tri):
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()


# This function generates a list of triangle object from Delaunay object
# returns a list of triangles
def triangleFromDelaunay(delaunyObject, points):
    triangles = []
    for each in delaunyObject.simplices:
        newTri = Triangle(np.array([points[each[0]], points[each[1]], points[each[2]]]))
        triangles.append(newTri)
    return triangles


# This function read points from file
# returns a n X 2 numpy array
def pointsFromFile(filePath):
    with open(filePath) as file:
        lines = file.readlines()
    result = []
    for eachLine in lines:
        data = eachLine.split()
        data = [float(data[0]), float(data[1])]
        result.append(data)
    return np.array(result)


class Triangle:
    # vertices is a 3 X 2 np array with datatype float64
    def __init__(self, vertices):
        if vertices.dtype is not np.dtype('float64'):
            raise ValueError('Vertices of the triangle should have a type of float64')
        self.vertices = vertices

    # iterate over the vertices of the triangle
    def __iter__(self):
        return iter(self.vertices)

    # For debug purpose
    def __str__(self):
        return str(self.vertices)

    # check if a point is within the triangle
    # point is an list of [x, y]
    def __contains__(self, point):
        area = getArea(self.vertices[0], self.vertices[1], self.vertices[2])
        area1 = getArea(point, self.vertices[1], self.vertices[2])
        area2 = getArea(self.vertices[0], point, self.vertices[2])
        area3 = getArea(self.vertices[0], self.vertices[1], point)
        return area == area1 + area2 + area3

    # Returns an n × 2 numpy array, of type float64, containing the (x, y)
    # coordinates of all points with integral
    def getPoints(self):
        # [0, 0] is the origin at upper left corner
        upperLeft = [sys.maxsize, sys.maxsize]  # upper left corner of the rectangle that contains the triangle
        lowerRight = [0, 0]  # lower right corner of the rectangle that contains the triangle
        # find a rectangle that contains the triangle
        for each in self:
            if each[0] < upperLeft[0]:
                upperLeft[0] = each[0]
            if each[1] < upperLeft[1]:
                upperLeft[1] = each[1]
            if each[0] > lowerRight[0]:
                lowerRight[0] = each[0]
            if each[1] > lowerRight[1]:
                lowerRight[1] = each[1]
        # check which points inside the rectangle are in the triangle
        result = []
        for eachX in range(floor(upperLeft[0]), ceil(lowerRight[0] + 1)):
            for eachY in range(floor(upperLeft[1]), ceil(lowerRight[1] + 1)):
                if [eachX, eachY] in self:
                    result.append([eachX, eachY])
        return np.array(result)


# Takes three points each as an iterable
# Returns the area formed by three points
def getArea(p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])) / 2.0)


if __name__ == '__main__':
    leftFile = 'points.left.txt'
    rightFile = 'points.right.txt'
    (leftTri, rightTri) = loadTriangles(leftFile, rightFile)
    print(getArea([0, 0], [1, 0], [0, 1]))

    triangleTest = Triangle(np.array([[0, 2], [2.0, 0], [4, 2]]))
    triangleTest.getPoints()
    print(triangleTest.getPoints())
