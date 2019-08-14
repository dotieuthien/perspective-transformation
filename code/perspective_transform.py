import cv2
import numpy as np
from skimage import feature, color, transform, io
import matplotlib.pyplot as plt
import logging
import math
from pylsd.lsd import lsd


# Funcs
def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    # for i in range(lines.shape[0]):
    #     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    #     pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    #     width = lines[i, 4]
    #     cv2.line(image, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
    # cv2.imwrite('test_out.jpg', image)
    return lines


def get_perspective_matrix(theta, phi, gamma, dx, dy, dz, w, h, f):
    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, - w / 2],
                    [0, 1, - h / 2],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w /2, 0],
                    [0, f, h / 2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


class PerspectiveTransform:
    # def vanishing_points(self, image):
    #     # image = edge_detect(image)
    #     line_detect(image)

    def rotate_axis(self, image, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        h, w, _ = np.shape(image)
        # Get radius of rotation along 3 axes
        rtheta = theta * np.pi / 180
        rphi = phi * np.pi / 180
        rgamma = phi * np.pi / 180

        d = np.sqrt(h ** 2 + w ** 2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = focal
        mat = get_perspective_matrix(rtheta, rphi, rgamma, dx, dy, dz, w, h, focal)
        print(mat)
        return cv2.warpPerspective(image.copy(), mat, (w, h))


# Test
transform = PerspectiveTransform()
image = cv2.imread('../test_images/5.png')
# image = edge_detection(image, net)
# image = line_detection(image)

# # p.vanishing_points(image)
image_out = transform.rotate_axis(image, theta=20, phi=0, gamma=0, dx=-5, dy=0, dz=0)
plt.imshow(image_out)
plt.show()
