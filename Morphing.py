#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 13
# #######################################################
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import sys
from math import ceil, floor
import imageio


# This function takes in the full file paths of the text files containing the (x, y)
# coordinates of a list of points, for both the left and right images.
# returns the tuple (leftTriangles, rightTriangles), where each is a list of instances of the Triangle class.
def loadTriangles(leftPointFilePath, rightPointFilePath):
    leftPoints = pointsFromFile(leftPointFilePath)
    rightPoints = pointsFromFile(rightPointFilePath)
    leftDelaunay = Delaunay(leftPoints)
    leftTriangles = triangleFromDelaunay(leftDelaunay, leftPoints)
    rightTriangles = triangleFromDelaunay(leftDelaunay, rightPoints)
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


class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        if leftImage.dtype is not np.dtype('uint8'):
            raise TypeError('Image type error!')
        if rightImage.dtype is not np.dtype('uint8'):
            raise TypeError('Image type error!')
        triangleTypeCheck(leftTriangles)
        triangleTypeCheck(rightTriangles)
        self.leftImage = leftImage
        self.leftTriangles = leftTriangles
        self.rightImage = rightImage
        self.rightTriangles = rightTriangles

    def getImageAtAlpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha should be within [0, 1]')
        # generate middle triangle
        midTriangles = []
        for eachLeft, eachRight in zip(self.leftTriangles, self.rightTriangles):
            points = []
            for eachLeftPoint, eachRightPoint in zip(eachLeft, eachRight):
                x = eachLeftPoint[0] * (1 - alpha) + eachRightPoint[0] * alpha  # x coordinate of middle triangle
                y = eachLeftPoint[1] * (1 - alpha) + eachRightPoint[1] * alpha  # y coordinate of middle triangle
                points.append([x, y])
            newTri = Triangle(np.array(points))
            midTriangles.append(newTri)
        # create middle image and begin transformation process
        midImage = np.zeros(self.leftImage.shape)
        for eachLeft, eachRight, eachMid in zip(self.leftTriangles, self.rightTriangles, midTriangles):
            # calculate affine transformation matrix
            # create matrices for calculation
            A_matrixL = []
            A_matrixR = []
            b_matrix = []
            for eachLP, eachRP, eachMP in zip(eachLeft, eachRight, eachMid):
                A_matrixL.append([eachLP[0], eachLP[1], 1, 0, 0, 0])
                A_matrixL.append([0, 0, 0, eachLP[0], eachLP[1], 1])
                A_matrixR.append([eachRP[0], eachRP[1], 1, 0, 0, 0])
                A_matrixR.append([0, 0, 0, eachRP[0], eachRP[1], 1])
                b_matrix.append([eachMP[0]])
                b_matrix.append([eachMP[1]])
            A_matrixL = np.array(A_matrixL)
            A_matrixR = np.array(A_matrixR)
            b_matrix = np.array(b_matrix)
            # solve for h in (6,1) size
            h_matrixL = np.linalg.solve(A_matrixL, b_matrix)
            h_matrixR = np.linalg.solve(A_matrixR, b_matrix)
            # rearrange to get affine projection matrix
            h_matrixL = rearrangeH(h_matrixL)
            h_matrixR = rearrangeH(h_matrixR)


# rearrange calculated h matrix form (6,1) to (3,3)
# returns affine projection matrix, in format:
# [h11 h12 h13
#  h21 h22 h23
#  0   0   1  ]
def rearrangeH(h):
    result = h.reshape(2, 3)
    result = np.concatenate((result, np.array([[0, 0, 1]])), axis=0)
    return result


# Checks if the input is a list of triangles
def triangleTypeCheck(triangles):
    for each in triangles:
        if type(each) is not Triangle:
            raise TypeError('input must be list of triangles')


if __name__ == '__main__':
    leftFile = 'points.left.txt'
    rightFile = 'points.right.txt'
    (leftTri, rightTri) = loadTriangles(leftFile, rightFile)
    # print(getArea([0, 0], [1, 0], [0, 1]))

    triangleTest = Triangle(np.array([[0, 2], [2.0, 0], [4, 2]]))
    triangleTest.getPoints()
    # print(triangleTest.getPoints())

    leftImage_test = imageio.imread('LeftGray.png')
    rightImage_test = imageio.imread('RightGray.png')
    # print(leftImage_test[4][1])
    # print(map_coordinates(leftImage_test, [[1],[1]]))
    morpher_test = Morpher(leftImage_test, leftTri, rightImage_test, rightTri)
    morpher_test.getImageAtAlpha(0.25)
