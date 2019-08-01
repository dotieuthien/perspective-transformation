import os
import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import inv


# Funcs for paper "rectification of planar targets using line segments"

def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(image, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
    cv2.imwrite('test_out.jpg', image)
    return lines


def cost_function(image, lines, theta=0, phi=0, gamma=0, f=1, epsilon=1):
    h, w, _ = np.shape(image)

    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    
    RY = np.array([[np.cos(phi), 0, -np.sin(phi)],
                   [0, 1, 0],
                   [np.sin(phi), 0, np.cos(phi)]])
    
    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)
    t = [0, 0, -max(h, w)]
    R[:, 2] = t
    # H function
    H = np.dot(K, R)
    H1 = inv(H)

    # The 1st term
    E = 0
    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        w = (line[0] + line[2])**2 + (line[1] + line[3])**2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2
        E = E + w * d
    
    # The 2nd term
    a = max(h, w)
    F = (max(a, f) / min(a, f) -1) ** 2

    retification = E + epsilon * F
    return retification


def derivative_d():
    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    
    RY = np.array([[np.cos(phi), 0, -np.sin(phi)],
                   [0, 1, 0],
                   [np.sin(phi), 0, np.cos(phi)]])
    
    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    pass


def optimizer():
    pass


# Test
image = cv2.imread('test_images/4.jpg')
lines = line_detection(image)
retification = cost_function(image, lines)
print(retification)