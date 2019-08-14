import os
import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import inv
from sympy import *
from skimage.transform import warp


# Funcs for paper "rectification of planar targets using line segments"

def line_detection(image):
    h, w, _ = np.shape(image)
    image_one = np.ones(np.shape(image)) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    center_x = h / 2
    center_y = w / 2
    p_lines = []
    for line in lines:
        x1 = line[0] - center_x
        y1 = center_y - line[1]

        x2 = line[2] - center_x
        y2 = center_y - line[3]
        p_line = [x1, y1, x2, y2, line[4]]
        p_lines.append(p_line)

    # for i in range(lines.shape[0]):
    #     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    #     pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    #     width = lines[i, 4]
    #     cv2.line(image_one, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
    # cv2.imwrite('test_out.jpg', image_one)

    return p_lines


def cost_function(h, w, lines, theta=0, phi=0, gamma=0, f=1, epsilon=0.1):
    K = np.array([[f, 0, w / 2],
                  [0, f,  h / 2],
                  [0, 0, 1]])
    
    K1 = np.array([[1 / f, 0, - w / (2*f)],
                   [0, 1 / f, - h / (2*f)],
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

    # H function
    H1 = np.dot(K, np.dot(R, K1))

    # The 1st term
    E = 0

    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        # weighted factor for cost
        w = (line[0] + line[2])**2 + (line[1] + line[3])**2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2]))**2
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

    K1 = np.array([[1 / f, 0, - w / (2*f)],
                   [0, 1 / f, - h / (2*f)],
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

    # Inverse response funtion H
    R1 = R.T
    H1 = K.dot(R.dot(K1))
    # H1 = 1 / h * R1.dot(K1)

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

    K1 = np.array([[1 / f_, 0, - w_ / (2*f_)],
                   [0, 1 / f_, - h_ / (2*f_)],
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
    R = np.dot(np.dot(RX, RZ), RY)

    # H -1 function
    H1 = np.dot(K, np.dot(R, K1))
    # H1 = 1 / h_ * np.dot(R.T, inv(K))

    # Reset gradient
    dE1 = 0
    dE2 = 0
    dE3 = 0
    dE4 = 0

    e = []

    # Finding the potential inlier
    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        d = (u1[0] / u1[2] - v1[0] / v1[2])**2 + (u1[1] / u1[2] - v1[1] / v1[2])**2
        du = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2

        e.append(du / d)

    mean_e = np.mean(e)
    std_e = np.std(e)
    ti = max(np.sin(np.pi / 60), min(mean_e + 2 * std_e, np.sin(np.pi / 10)))

    count = -1
    count_ = 0

    if f_ < h_:
        dF = - h_ / (f_**2)
    elif f_ > h_:
        dF = 1 / h_
    else:
        dF = 0

    for line in lines:
        count += 1
        if e[count] > ti:
            continue
        count_ += 1
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)

        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        # derivative for E (1st term)
        weighted = (line[0] - line[2])**2 + (line[1] - line[3])**2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2

        if (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] > v1[0] / v1[2]):
            dd_H1u = np.array([1 / u1[2], -1 / u1[2], (-u1[0] + u1[1]) / (u1[2] ** 2)])
            dE_tmp = weighted * 2 * np.sqrt(d) * dd_H1u
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
            dE4 = dE4 + dF + dE_tmp.dot(H1u_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] <= v1[0] / v1[2]):
            dd_H1u = np.array([-1 / u1[2], 1 / u1[2], (u1[0] - u1[1]) / (u1[2] ** 2)])
            dE_tmp = weighted * 2 * np.sqrt(d) * dd_H1u
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
            dE4 = dE4 + dF + dE_tmp.dot(H1u_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) > abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[1] / u1[2] <= v1[1] / v1[2]):
            dd_H1v = np.array([1 / v1[2], -1 / v1[2], (-v1[0] + v1[1]) / (v1[2] ** 2)])
            dE_tmp = weighted * 2 * np.sqrt(d) * dd_H1v
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
            dE4 = dE4 + dF + dE_tmp.dot(H1v_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

        else:
            dd_H1v = np.array([-1 / v1[2], 1 / v1[2], (v1[0] - v1[1]) / (v1[2] ** 2)])
            dE_tmp = weighted * 2 * np.sqrt(d) * dd_H1v
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
            dE4 = dE4 + dF + dE_tmp.dot(H1v_f.subs({f:f_, theta:theta_, phi:phi_, gamma:gamma_, u1:u[0], u2:u[1], u3:u[2],
                                                 v1:v[0], v2:v[1], v3:v[2], h:h_, w: w_}))

    dE = [dE1 / count_, dE2 / count_, dE3 / count_, dE4 / count_]
    return dE


def gradient_descent_optimizer(image, max_iters, learning_rate):
    init_f = 250
    init_theta = 0
    init_phi = 0
    init_gamma = 0

    h, w, _ = np.shape(image)
    # Feed forward
    lines = line_detection(image)

    # Cost function
    rectification = cost_function(h, w, lines, init_theta, init_phi, init_gamma, init_f)
    print('Cost Value: ', rectification)

    for i in range(max_iters):
        # Compute gradient
        theta, phi, gamma, f = gradient(image, lines, init_f, init_theta, init_phi, init_gamma)

        # Process the angel
        if theta % (2 * np.pi) > np.pi:
            theta = - 2 * np.pi + (theta % (2 * np.pi))
        else:
            theta = theta % (2 * np.pi)

        if phi % (2 * np.pi) > np.pi:
            phi = - 2 * np.pi + (phi % (2 * np.pi))
        else:
            phi = phi % (2 * np.pi)

        if gamma % (2 * np.pi) > np.pi:
            gamma = - 2 * np.pi + (gamma % (2 * np.pi))
        else:
            gamma = gamma % (2 * np.pi)
        
        print('Gradient ', theta, phi, gamma)

        init_f = np.float32((init_f))
        init_theta = np.float32(- theta * 0.01)
        init_phi = np.float32(- phi * 0.01)
        init_gamma = np.float32(- gamma * 0.01)

        rectification = cost_function(h, w, lines, init_theta, init_phi, init_gamma, init_f)
        print('Cost Value: ', rectification)

        K = np.array([[init_f, 0, w / 2],
                      [0, init_f, h / 2],
                      [0, 0, 1]])

        K1 = np.array([[1 / init_f, 0, - w / (2 * init_f)],
                       [0, 1 / init_f, - h / (2 * init_f)],
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

        # H -1 function
        H1 = np.dot(K, np.dot(R, K1))

        image = cv2.warpPerspective(image.copy(), np.float32(H1), (w, h))
        cv2.imwrite('image_out_.png', image)



if __name__ == '__main__':
    image = cv2.imread('../test_images/1.jpg')
    gradient_descent_optimizer(image, 30, 0.1)