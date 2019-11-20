#######################################################
# Author:   Zhengsen Fu
# email:    fu216@purdue.edu
# ID:       0029752483
# Date:     Nov 13
# #######################################################
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
# import sys
from math import ceil, floor
import imageio
import tempfile
import os
import ffmpeg
from pprint import pprint as pp


# This function takes in the full file paths of the text files containing the (x, y)
# coordinates of a list of points, for both the left and right images.
# returns the tuple (leftTriangles, rightTriangles), where each is a list of instances
# of the Triangle class.
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
    # slower implementation
    # def getPoints(self):
    #
    #     # [0, 0] is the origin at upper left corner
    #     upperLeft = [sys.maxsize, sys.maxsize]  # upper left corner of the rectangle
    #     that contains the triangle
    #     lowerRight = [0, 0]  # lower right corner of the rectangle that contains the triangle
    #     # find a rectangle that contains the triangle
    #     for each in self:
    #         if each[0] < upperLeft[0]:
    #             upperLeft[0] = each[0]
    #         if each[1] < upperLeft[1]:
    #             upperLeft[1] = each[1]
    #         if each[0] > lowerRight[0]:
    #             lowerRight[0] = each[0]
    #         if each[1] > lowerRight[1]:
    #             lowerRight[1] = each[1]
    #     # check which points inside the rectangle are in the triangle
    #     result = []
    #     for eachX in range(floor(upperLeft[0]), ceil(lowerRight[0] + 1)):
    #         for eachY in range(floor(upperLeft[1]), ceil(lowerRight[1] + 1)):
    #             if [eachX, eachY] in self:
    #                 result.append([eachX, eachY])
    #     return np.array(result)

    def getPoints(self):
        # ordered as column-row  plane
        points = sorted([list(x) for x in self], key=order)
        upper = points[0]
        middle = points[1]
        lower = points[2]
        upperLower = equationCalc(upper, lower)
        upperMiddle = equationCalc(upper, middle)
        middleLower = equationCalc(middle, lower)
        points = []
        # Find points in the upper part
        for y in range(ceil(upper[1]), floor(middle[1])):
            a = upperLower(y)
            b = upperMiddle(y)
            for x in range(ceil(min(a, b)), floor(max(a, b)) + 1):
                points.append([x, y])
        # Find points in the lower part
        for y in range(floor(middle[1]), floor(lower[1] + 1)):
            a = upperLower(y)
            b = middleLower(y)
            for x in range(ceil(min(a, b)), floor(max(a, b)) + 1):
                points.append([x, y])
        return np.array(points)


# calculate linear equation with two points provided
# calculate m and b value
# x = my + b
# return a function takes input y and output corresponding x
def equationCalc(a, b):
    if (a[1] - b[1]) != 0:
        m = (a[0] - b[0]) / (a[1] - b[1])
    else:
        m = 0
    b = a[0] - m * a[1]

    # take y value and return x value
    def F(y):
        return m * y + b

    return F


# provide order for sorted() in getPoints(self)
def order(x):
    return x[1]


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

    # Generates triangles for middle image
    # return a list of triangles like leftTriangles
    # called by getImageAtAlpha(self, alpha)
    def _generateMiddleTri(self, alpha):
        midTriangles = []
        for eachLeft, eachRight in zip(self.leftTriangles, self.rightTriangles):
            points = []
            for eachLeftPoint, eachRightPoint in zip(eachLeft, eachRight):
                # x coordinate of middle triangle
                x = eachLeftPoint[0] * (1 - alpha) + eachRightPoint[0] * alpha
                # y coordinate of middle triangle
                y = eachLeftPoint[1] * (1 - alpha) + eachRightPoint[1] * alpha
                points.append([x, y])
            newTri = Triangle(np.array(points))
            midTriangles.append(newTri)
        return midTriangles

    def getImageAtAlpha(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha should be within [0, 1]')

        # generate middle triangle
        midTriangles = self._generateMiddleTri(alpha)
        # create middle image and begin transformation process
        midImage = np.zeros(self.leftImage.shape)
        leftInterp = RectBivariateSpline(range(self.leftImage.shape[0]),
                                         range(self.leftImage.shape[1]),
                                         self.leftImage,
                                         kx=1, ky=1)
        rightInterp = RectBivariateSpline(range(self.rightImage.shape[0]),
                                          range(self.rightImage.shape[1]),
                                          self.rightImage,
                                          kx=1, ky=1)

        for eachLeft, eachRight, eachMid in zip(self.leftTriangles, self.rightTriangles, midTriangles):
            # calculate affine transformation matrix
            h_matrixL, h_matrixR = hMatrixCalc(eachLeft, eachRight, eachMid)
            h_inverseL = np.linalg.inv(h_matrixL)
            h_inverseR = np.linalg.inv(h_matrixR)
            # fill the middle image with affine blend
            # find points within middle triangle
            points = eachMid.getPoints()
            # check for items out of bound
            points = checkBoundary(points, self.leftImage.shape[1], self.leftImage.shape[0])
            # for index, each in enumerate(points):
            #     if each[0] >= self.leftImage.shape[1] or each[1] >= self.leftImage.shape[0]:
            #         print(each, index)
            #         points = np.delete(points, index, axis=0)
            #         printPoint(points)
            #         print()
            # insert 1 and transpose to make the become array of vertical matrix
            # for faster matrix operation
            points_matrix = np.insert(points, 2, 1, axis=1).T
            leftPoint = np.matmul(h_inverseL, points_matrix)  # [0]: x ; [1]: y
            rightPoint = np.matmul(h_inverseR, points_matrix)
            # midImage[points[:, 1], points[:, 0]] = (1 - alpha) * leftInterp.ev(leftPoint[1], leftPoint[0])
            # midImage[points[:, 1], points[:, 0]] += alpha * rightInterp.ev(rightPoint[1], rightPoint[0])
            try:
                midImage[points[:, 1], points[:, 0]] = (1 - alpha) * leftInterp.ev(leftPoint[1], leftPoint[0])
                midImage[points[:, 1], points[:, 0]] += alpha * rightInterp.ev(rightPoint[1], rightPoint[0])
            except IndexError:
                # print()
                for index, each in enumerate(points):
                    print(each, index)
                    print(each[0] >= self.leftImage.shape[1])
        midImage = midImage.astype(np.uint8)
        return midImage

    def saveVideo(self, targetFilePath, frameCount, frameRate, includeReversed):
        if frameCount < 10:
            raise ValueError('frameCount must be greater than 10')
        tempDir = tempfile.TemporaryDirectory()
        alphaIncrement = 1 / (frameCount - 1)
        for index in range(frameCount):
            try:
                alpha = index * alphaIncrement
                image = self.getImageAtAlpha(alpha)
                path = os.path.join(tempDir.name, f'{index}.png')
                imageio.imwrite(path, image)
                print('generating:', alpha)
            except IndexError:
                print(alpha)
        if includeReversed is False:
            (
                ffmpeg
                .input(os.path.join(tempDir.name, '*.png'), pattern_type='glob', framerate=frameRate)
                .output(targetFilePath)
                .run()
            )
        else:
            tempVideoPath = os.path.join(tempDir.name, 'temp.mp4')
            (
                ffmpeg
                .input(os.path.join(tempDir.name, '*.png'), pattern_type='glob', framerate=frameRate)
                .output(tempVideoPath)
                .run()
            )
            in1 = ffmpeg.input(tempVideoPath)
            in2 = ffmpeg.input(tempVideoPath)
            v2 = in2.video.filter('reverse')
            joined = ffmpeg.concat(in1, v2).node
            out = ffmpeg.output(joined, targetFilePath)
            out.run()


# takes points, self.leftImage.shape[1], self.leftImage.shape[0]
# recursive call to manipulate array of points
# to remove points that are out of bounds
# return np.array of points
def checkBoundary(points, boundaryX, boundaryY):
    for index, each in enumerate(points):
        if each[0] >= boundaryX or each[1] >= boundaryY:
            points = np.delete(points, index, axis=0)
            points = checkBoundary(points, boundaryX, boundaryY)
            break
    return points


def printPoint(points):
    for index, each in enumerate(points):
        print(index, ':', each)


# takes point as a np array of x and y coordinate
# image from imageio as a np array
# alpha as alpha blending parameter
# returns blended grey scale value
# def alphaBlend(point, image, alpha):
#     x = point[0]
#     y = point[1]
#     x1 = floor(x)
#     x2 = ceil(x)
#     y1 = ceil(y)
#     y2 = floor(y)
#     if y2 == y1 and y1 == 0:
#         y1 = y1 + 1
#     elif y1 == image.shape[0]:
#         y2 -= 1
#         y1 -= 1
#     elif y1 == y2:
#         y2 -= 1
#     if x == x1 and x == 0:
#         x2 = x2 + 1
#     elif x2 == image.shape[1]:
#         x1 -= 1
#         x2 -= 1
#     elif x1 == x2:
#         x1 -= 1
#     upperLeft = image[y2][x1]
#     upperRight = image[y2][x2]
#     lowerLeft = image[y1][x1]
#     lowerRight = image[y1][x2]
#     interpolated = (x2 - x) * (y1 - y) * upperLeft
#     interpolated += (x - x1) * (y1 - y) * upperRight
#     interpolated += (x2 - x) * (y - y2) * lowerLeft
#     interpolated += (x - x1) * (y - y2) * lowerRight
#     return interpolated * alpha


# take in a point in middle triangle and inverse H matrix
# to map the point back to original image
# return x and y coordinates as a list
# called by getImageAtAlpha(self, alpha)
def affineTransform(point, matrix):
    pMatrix = point.reshape(2, 1)
    pMatrix = np.append(pMatrix, [[1]], axis=0)
    mapped = np.matmul(matrix, pMatrix)
    result = [mapped[0][0], mapped[1][0]]
    return result


# takes in triangles from left image, right image, and generated middle image
# returns affine transformation matrices for both left image and right image
# called by getImageAtAlpha(self, alpha)
def hMatrixCalc(left, right, mid):
    # create matrices for calculation
    A_matrixL = []
    A_matrixR = []
    b_matrix = []
    for eachLP, eachRP, eachMP in zip(left, right, mid):
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
    return h_matrixL, h_matrixR


# rearrange calculated h matrix form (6,1) to (3,3)
# returns affine projection matrix, in format:
# [h11 h12 h13
#  h21 h22 h23
#  0   0   1  ]
# called byhMatrixCalc(left, right,  mid)
def rearrangeH(h):
    result = h.reshape(2, 3)
    result = np.concatenate((result, np.array([[0, 0, 1]])), axis=0)
    return result


# Checks if the input is a list of triangles
# called by getImageAtAlpha(self, alpha)
def triangleTypeCheck(triangles):
    for each in triangles:
        if type(each) is not Triangle:
            raise TypeError('input must be list of triangles')


if __name__ == '__main__':
    leftFile = 'points.left.txt'
    rightFile = 'points.right.txt'
    (leftTri, rightTri) = loadTriangles(leftFile, rightFile)
    # print(getArea([0, 0], [1, 0], [0, 1]))

    triangleTest = Triangle(np.array([[1439.0, 0], [853.2, 619.2], [1171.8, 507.6]]))
    # print(triangleTest.getPoints().sort() == triangleTest.getPoints2().sort())
    # triangleTest.getPoints2()

    leftImage_test = imageio.imread('LeftGray.png')
    rightImage_test = imageio.imread('RightGray.png')
    # print(leftImage_test[4][1])
    # print(map_coordinates(leftImage_test, [[1],[1]]))
    # print(leftImage_test.shape)
    morpher_test = Morpher(leftImage_test, leftTri, rightImage_test, rightTri)
    morphed = morpher_test.getImageAtAlpha(1/9)
    imageio.imwrite('result.png', morphed)
    # # print(morphed[187][404])
    # plt.imshow(morphed)
    # plt.show()

    # point_test = np.array([0.5, 1.5])
    # matrix_test = np.array([[1,2,3],
    #                         [4,5,6],
    #                         [0,0,1]])
    # affineTransform(point_test, matrix_test)
    #
    # print(alphaBlend(point_test,matrix_test,1))
    # tempDir_test = tempfile.TemporaryDirectory()
    # print(tempDir_test.name)
    # image_test = os.path.join(tempDir_test.name, '1.png')
    # print(image_test)
    morpher_test.saveVideo('out.mp4', 10, 5, False)
