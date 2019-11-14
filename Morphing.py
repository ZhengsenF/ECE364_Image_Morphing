#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 13
# #######################################################
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


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


# This function generates a list of triangle object from Delaunay object
# returns a list of triangles
def triangleFromDelaunay(delaunyObject, points):
    triangles = []
    for each in delaunyObject.simplices:
        newTri = Triangle(np.array([points[each[0]], points[each[1]], points[each[2]]]))
        triangles.append(newTri)
        print(newTri)
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

    # For debug purpose
    def __str__(self):
        return str(self.vertices)

    def getPoints(self):
        pass


if __name__ == '__main__':
    leftFile = 'points.left.txt'
    rightFile = 'points.right.txt'
    (leftTri, rightTri) = loadTriangles(leftFile, rightFile)
