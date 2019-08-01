import cv2
import numpy as np
from skimage import feature, color, transform, io
import matplotlib.pyplot as plt
import logging
import math


# Funcs
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
 
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
 
        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batchSize, numChannels, height, width]]
 
    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv2.dnn_registerLayer('Crop', CropLayer)
net = cv2.dnn.readNet('edge_detection/deploy.prototxt', 'edge_detection/hed_pretrained_bsds.caffemodel')


def edge_detection(image, net):
    h, w, _ = np.shape(image)
    inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (w, h))
    out = 255 * out
    out = out.astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out
    

def line_detection(image, sigma=3):
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=2)
    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # Convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    orientation = []

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for i in range(locations.shape[0]):
        # Horizontal
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]

        # Vectical
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        orientation = (yax[1] - yax[0]) / math.sqrt((xax[1] - xax[0])**2 + (yax[1] - yax[0])**2)
        angel = 180 - (math.acos(orientation) * 180 / math.pi)
        if angel > -10 and angel < 10:
            plt.plot(xax, yax, 'r-')

    plt.show()
    return edges


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
    A2 = np.array([ [f, 0, w / 2, 0],
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
        return cv2.warpPerspective(image.copy(), mat, (w, h))


# Test
transform = PerspectiveTransform()
image = cv2.imread('test_images/4.jpg')
# image = edge_detection(image, net)
# image = line_detection(image)

# # p.vanishing_points(image)
image_out = transform.rotate_axis(image, theta=30, dx=5)
plt.imshow(image_out)
plt.show()
