import os
import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import inv
from sympy import *


# Funcs for paper "rectification of planar targets using line segments"

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


def cost_function(image, lines, theta=0, phi=0, gamma=0, f=1, epsilon=1):
    h, w, _ = np.shape(image)

    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
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

    t = [0, 0, 1]
    R[:, 2] = t

    # H function
    H = np.dot(K, np.dot(R, inv(K)))
    # H1 = np.dot(K, np.dot(R, inv(K)))
    H1 = H

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

    rectification = E + epsilon * F
    return rectification


def derivative_componets():
    # Build graph of model
    f, theta, phi, gamma = symbols('f theta phi gamma')
    u1, u2, u3 = symbols('u1 u2 u3')
    v1, v2, v3 = symbols('v1 v2 v3')
    w, h = symbols('w h')

    # K matrix for internal paprameter of camera
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]])

    K1 = np.array([[1/f, 0, - w / 2],
                  [0, 1/f, - h / 2],
                  [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0],
                   [0, cos(theta), -sin(theta)],
                   [0, sin(theta), cos(theta)]])
    
    RY = np.array([[cos(phi), 0, -sin(phi)],
                   [0, 1, 0],
                   [sin(phi), 0, cos(phi)]])
    
    RZ = np.array([[cos(gamma), -sin(gamma), 0],
                   [sin(gamma), cos(gamma), 0],
                   [0, 0, 1]])

    RXY = RX.dot(RY)
    # Rotation matrix
    R = RXY.dot(RZ)

    # Translation vector
    t = [0, 0, 1]
    R[:, 2] = t
    # Inverse response funtion H
    # R1 = Matrix(R).T
    H1 = K.dot(R.dot(K1))

    # Compute derivative for each component
    u = Matrix([u1, u2, u3])
    H1u = H1.dot(u)
    H1u_theta = diff(H1u, theta)
    H1u_phi = diff(H1u, phi)
    H1u_gamma = diff(H1u, gamma)
    H1u_f = diff(H1u, f)

    v = Matrix([v1, v2, v3])
    H1v = H1.dot(v)
    H1v_theta = diff(H1v, theta)
    H1v_phi = diff(H1v, phi)
    H1v_gamma = diff(H1v, gamma)
    H1v_f = diff(H1v, f)
    return H1u_theta, H1u_phi, H1u_gamma, H1u_f, H1v_theta, H1v_phi, H1v_gamma, H1v_f


def gradient(image, lines, f_, theta_, phi_, gamma_):
    H1u_theta, H1u_phi, H1u_gamma, H1u_f, H1v_theta, H1v_phi, H1v_gamma, H1v_f = derivative_componets()

    h_, w_, _ = np.shape(image)

    K = np.array([[f_, 0, w_ / 2],
                  [0, f_, h_ / 2],
                  [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0],
                   [0, np.cos(theta_), -np.sin(theta_)],
                   [0, np.sin(theta_), np.cos(theta_)]])
    
    RY = np.array([[np.cos(phi_), 0, -np.sin(phi_)],
                   [0, 1, 0],
                   [np.sin(phi_), 0, np.cos(phi_)]])
    
    RZ = np.array([[np.cos(gamma_), -np.sin(gamma_), 0],
                   [np.sin(gamma_), np.cos(gamma_), 0],
                   [0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)
    t = [0, 0, 1]
    R[:, 2] = t
    # H function
    H = np.dot(K, np.dot(R, inv(K)))
    H1 = H

    dE1 = 0
    dE2 = 0
    dE3 = 0
    dE4 = 0

    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        # derivative for E (1st term)
        weighted = (line[0] + line[2])**2 + (line[1] + line[3])**2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2

        if (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] > v1[0] / v1[2]):
            dd_H1u = np.array([1 / u1[2], -1 / u1[2], (-u1[0] + u1[1]) / (u1[2] ** 2)])
            dE_tmp = weighted * 2 * d * dd_H1u
            f, theta, phi, gamma = symbols('f theta phi gamma')
            u1, u2, u3 = symbols('u1 u2 u3')
            v1, v2, v3 = symbols('v1 v2 v3')
            w, h = symbols('w h')

            dE1 = dE1 + dE_tmp.dot(H1u_theta.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE2 = dE2 + dE_tmp.dot(H1u_phi.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE3 = dE3 + dE_tmp.dot(H1u_gamma.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE4 = dE4 + dE_tmp.dot(H1u_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] <= v1[0] / v1[2]):
            dd_H1u = np.array([-1 / u1[2], 1 / u1[2], (u1[0] - u1[1]) / (u1[2] ** 2)])
            dE_tmp = weighted * 2 * d * dd_H1u
            f, theta, phi, gamma = symbols('f theta phi gamma')
            u1, u2, u3 = symbols('u1 u2 u3')
            v1, v2, v3 = symbols('v1 v2 v3')
            w, h = symbols('w h')

            dE1 = dE1 + dE_tmp.dot(H1u_theta.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE2 = dE2 + dE_tmp.dot(H1u_phi.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE3 = dE3 + dE_tmp.dot(H1u_gamma.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE4 = dE4 + dE_tmp.dot(H1u_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) > abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[1] / u1[2] <= v1[1] / v1[2]):
            dd_H1v = np.array([1 / v1[2], -1 / v1[2], (-v1[0] + v1[1]) / (v1[2] ** 2)])
            dE_tmp = weighted * 2 * d * dd_H1v
            f, theta, phi, gamma = symbols('f theta phi gamma')
            u1, u2, u3 = symbols('u1 u2 u3')
            v1, v2, v3 = symbols('v1 v2 v3')
            w, h = symbols('w h')

            dE1 = dE1 + dE_tmp.dot(H1v_theta.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE2 = dE2 + dE_tmp.dot(H1v_phi.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE3 = dE3 + dE_tmp.dot(H1v_gamma.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE4 = dE4 + dE_tmp.dot(H1v_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        else:
            dd_H1v = np.array([-1 / v1[2], 1 / v1[2], (v1[0] - v1[1]) / (v1[2] ** 2)])
            dE_tmp = weighted * 2 * d * dd_H1v
            f, theta, phi, gamma = symbols('f theta phi gamma')
            u1, u2, u3 = symbols('u1 u2 u3')
            v1, v2, v3 = symbols('v1 v2 v3')
            w, h = symbols('w h')

            dE1 = dE1 + dE_tmp.dot(H1v_theta.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE2 = dE2 + dE_tmp.dot(H1v_phi.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE3 = dE3 + dE_tmp.dot(H1v_gamma.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))
            dE4 = dE4 + dE_tmp.dot(H1v_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

    dE = [dE1, dE2, dE3, dE4]
    return dE


def gradient_descent_optimizer(image, max_iters, learning_rate):
    init_f = 100
    init_theta = 0
    init_phi = 0
    init_gamma = 0

    for i in range(max_iters):
        # Feed forward
        lines = line_detection(image)
        # Loss function
        rectification = cost_function(image, lines)
        print(rectification)
        # Compute gradient
        f, theta, phi, gamma = gradient(image, lines, init_f, init_theta, init_phi, init_gamma)
        print(theta, phi, gamma)
        init_f = np.float32(398)
        init_theta = np.float32(init_theta - theta * np.pi / 180)
        init_phi = np.float32(init_phi - phi * np.pi / 180)
        init_gamma = np.float32(init_gamma - gamma * np.pi / 180)

        # init_theta, init_phi, init_gamma = -20, 0, 0
        # init_theta = np.float32(init_theta * np.pi / 180)
        # init_phi = np.float32(init_phi * np.pi / 180)
        # init_gamma = np.float32(init_gamma * np.pi / 180)

        h, w, _ = np.shape(image)

        K = np.array([[init_f, 0, w /2],
                    [0, init_f, h / 2],
                    [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0],
                    [0, np.cos(init_theta), -np.sin(init_theta)],
                    [0, np.sin(init_theta), np.cos(init_theta)]])
        
        RY = np.array([[np.cos(init_phi), 0, -np.sin(init_phi)],
                    [0, 1, 0],
                    [np.sin(init_phi), 0, np.cos(init_phi)]])
        
        RZ = np.array([[np.cos(init_gamma), -np.sin(init_gamma), 0],
                    [np.sin(init_gamma), np.cos(init_gamma), 0],
                    [0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)
        t = [0, 0, 1]
        R[:, 2] = t

        # H function
        H = np.dot(K, np.dot(R, inv(K)))
        H1 = H
        print(H1)
        image_out = cv2.warpPerspective(image.copy(), np.float32(H1), (w, h))
        cv2.imwrite('image_out_' + str(i) + '.png', image_out)



if __name__ == '__main__':
    image = cv2.imread('test_images/2.jpg')
    # lines = line_detection(image)
    # retification = cost_function(image, lines)
    # derivative_componets()
    gradient_descent_optimizer(image, 10, 0.1)