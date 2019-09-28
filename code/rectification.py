import os
import cv2
from pylsd.lsd import lsd
import numpy as np
from scipy.linalg import expm
import time


# Funcs for paper "rectification of planar targets using line segments"
def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray)
    p_lines = []
    image_one = np.ones(np.shape(image)) * 255

    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = np.sqrt((lines[i, 0] - lines[i, 2]) ** 2 + (lines[i, 1] - lines[i, 3]) ** 2)
        angle = abs(lines[i, 0] - lines[i, 2]) / abs(lines[i, 1] - lines[i, 3])

        if width > 10 and (angle < 1):
            p_lines.append([lines[i, 0], lines[i, 1], lines[i, 2], lines[i, 3]])
            cv2.line(image_one, pt1, pt2, (0, 0, 255), int(2))

    cv2.imwrite('test_lines.jpg', image_one)
    return p_lines


def crop_black_border(image, points):
    print(points)
    x1 = int(max(points[0][1] / points[0][2], points[1][1] / points[1][2]))
    x2 = int(min(points[2][1] / points[2][2], points[3][1] / points[3][2]))
    y1 = int(max(points[0][0] / points[0][2], points[2][0] / points[2][2]))
    y2 = int(min(points[1][0] / points[1][2], points[3][0] / points[3][2]))
    image_out = image[x1:x2, y1:y2, :]
    cv2.imwrite('output.jpg', image_out)
    return image_out


def cost_function(h, w, f, lines, H1):
    # The 1st term
    E = 0

    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        # weighted factor for cost
        w = (line[0] + line[2]) ** 2 + (line[1] + line[3]) ** 2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2
        E = E + w * d

    # The 2nd term
    a = max(h, w)
    F = (max(a, f) / min(a, f) -1) ** 2

    cost = E + 0.1 * F

    return cost


def derivative_componets(h_, w_, f_, theta_, phi_, gamma_, u_, v_):
    # K matrix for internal paprameter of camera
    K = np.array([[f_, 0, w_ / 2],
                  [0, f_, h_/ 2],
                  [0, 0, 1]])

    K1 = np.array([[1 / f_, 0, - w_ / (2 * f_)],
                   [0, 1 / f_, - h_ / (2 * f_)],
                   [0, 0, 1]])

    # Rotation matrices
    R = np.array([[0, -gamma_, phi_],
                  [gamma_, 0, -theta_],
                  [-phi_, theta_, 0]])

    R = expm(R)

    # Compute derivative for each component
    R_theta = np.array([[0, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
    H1u_theta = np.dot(K, np.dot(R_theta, np.dot(R, np.dot(K1, u_))))
    H1v_theta = np.dot(K, np.dot(R_theta, np.dot(R, np.dot(K1, v_))))

    R_phi = np.array([[0, 0, 1],
                      [0, 0, 0],
                      [-1, 0, 0]])
    H1u_phi = np.dot(K, np.dot(R_phi, np.dot(R, np.dot(K1, u_))))
    H1v_phi = np.dot(K, np.dot(R_phi, np.dot(R, np.dot(K1, v_))))

    R_gamma = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])
    H1u_gamma = np.dot(K, np.dot(R_gamma, np.dot(R, np.dot(K1, u_))))
    H1v_gamma = np.dot(K, np.dot(R_gamma, np.dot(R, np.dot(K1, v_))))

    return H1u_theta, H1u_phi, H1u_gamma, H1v_theta, H1v_phi, H1v_gamma


def gradient(image, lines, f_, theta_, phi_, gamma_):
    h_, w_, _ = np.shape(image)

    K = np.array([[f_, 0, w_ / 2],
                  [0, f_, h_ / 2],
                  [0, 0, 1]])

    K1 = np.array([[1 / f_, 0, - w_ / (2 * f_)],
                   [0, 1 / f_, - h_ / (2 * f_)],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    R = np.array([[0, -gamma_, phi_],
                  [gamma_, 0, -theta_],
                  [-phi_, theta_, 0]])
    R = expm(R)
    H1 = np.dot(K, np.dot(R, K1))

    # Reset gradient
    dE1 = 0
    dE2 = 0
    dE3 = 0

    e = []

    # Finding the potential inlier
    for line in lines:
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)
        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        d = (u1[0] / u1[2] - v1[0] / v1[2]) ** 2 + (u1[1] / u1[2] - v1[1] / v1[2]) ** 2
        du = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2

        e.append(du / d)

    mean_e = np.mean(e)
    std_e = np.std(e)
    ti = max(np.sin(np.pi / 60), min(mean_e + 2 * std_e, np.sin(np.pi / 10)))

    count = -1
    count_ = 0

    # Compute gradient
    image_one_in = np.ones((h_, w_, 3)) * 255

    for line in lines:
        count += 1
        # Remove outliers
        if e[count] > ti:
            pt1 = (int(line[0]), int(line[1]))
            pt2 = (int(line[2]), int(line[3]))
            cv2.line(image_one_in, pt1, pt2, (0, 255, 0), int(2))
            cv2.imwrite('test_inliers.jpg', image_one_in)
            continue

        pt1 = (int(line[0]), int(line[1]))
        pt2 = (int(line[2]), int(line[3]))
        cv2.line(image_one_in, pt1, pt2, (0, 0, 255), int(2))
        cv2.imwrite('test_inliers.jpg', image_one_in)

        count_ += 1
        # Convert to world coordinates
        u = [line[0], line[1], 1]
        u1 = np.dot(H1, u)

        v = [line[2], line[3], 1]
        v1 = np.dot(H1, v)

        weighted = (line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2
        d = min(abs(u1[0] / u1[2] - v1[0] / v1[2]), abs(u1[1] / u1[2] - v1[1] / v1[2])) ** 2
        # Derivative
        H1u_theta, H1u_phi, H1u_gamma, H1v_theta, H1v_phi, H1v_gamma = derivative_componets(h_, w_, f_, theta_, phi_, gamma_, u1, v1)

        if (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] > v1[0] / v1[2]):
            dd_H1u = np.array([1 / u1[2], 0, -u1[0] / (u1[2] ** 2)])
            dE_tmp_u = weighted * 2 * np.sqrt(d) * dd_H1u

            dd_H1v = np.array([-1 / v1[2], 0, v1[0] / (v1[2] ** 2)])
            dE_tmp_v = weighted * 2 * np.sqrt(d) * dd_H1v

            dE1 = dE1 + np.dot(dE_tmp_u, H1u_theta) + np.dot(dE_tmp_v, H1v_theta)
            dE2 = dE2 + np.dot(dE_tmp_u, H1u_phi) + np.dot(dE_tmp_v, H1v_phi)
            dE3 = dE3 + np.dot(dE_tmp_u, H1u_gamma) + np.dot(dE_tmp_v, H1v_gamma)
            

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) <= abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[0] / u1[2] <= v1[0] / v1[2]):
            dd_H1u = np.array([-1 / u1[2], 0, u1[0] / (u1[2] ** 2)])
            dE_tmp_u = weighted * 2 * np.sqrt(d) * dd_H1u

            dd_H1v = np.array([1 / v1[2], 0, -v1[0] / (u1[2] ** 2)])
            dE_tmp_v = weighted * 2 * np.sqrt(d) * dd_H1v

            dE1 = dE1 + np.dot(dE_tmp_u, H1u_theta) + np.dot(dE_tmp_v, H1v_theta)
            dE2 = dE2 + np.dot(dE_tmp_u, H1u_phi) + np.dot(dE_tmp_v, H1v_phi)
            dE3 = dE3 + np.dot(dE_tmp_u, H1u_gamma) + np.dot(dE_tmp_v, H1v_gamma)
            

        elif (abs(u1[0] / u1[2] - v1[0] / v1[2]) > abs(u1[1] / u1[2] - v1[1] / v1[2])) and (u1[1] / u1[2] <= v1[1] / v1[2]):
            dd_H1u = np.array([0, -1 / u1[2], u1[1] / (u1[2] ** 2)])
            dE_tmp_u = weighted * 2 * np.sqrt(d) * dd_H1u

            dd_H1v = np.array([0, 1 / v1[2], -v1[1] / (v1[2] ** 2)])
            dE_tmp_v = weighted * 2 * np.sqrt(d) * dd_H1v

            dE1 = dE1 + np.dot(dE_tmp_u, H1u_theta) + np.dot(dE_tmp_v, H1v_theta)
            dE2 = dE2 + np.dot(dE_tmp_u, H1u_phi) + np.dot(dE_tmp_v, H1v_phi)
            dE3 = dE3 + np.dot(dE_tmp_u, H1u_gamma) + np.dot(dE_tmp_v, H1v_gamma)
            
        else:
            dd_H1u = np.array([0, 1 / u1[2], -u1[1] / (u1[2] ** 2)])
            dE_tmp_u = weighted * 2 * np.sqrt(d) * dd_H1u

            dd_H1v = np.array([0, -1 / v1[2], v1[1] / (v1[2] ** 2)])
            dE_tmp_v = weighted * 2 * np.sqrt(d) * dd_H1v

            dE1 = dE1 + np.dot(dE_tmp_u, H1u_theta) + np.dot(dE_tmp_v, H1v_theta)
            dE2 = dE2 + np.dot(dE_tmp_u, H1u_phi) + np.dot(dE_tmp_v, H1v_phi)
            dE3 = dE3 + np.dot(dE_tmp_u, H1u_gamma) + np.dot(dE_tmp_v, H1v_gamma)

    dE = [np.float32(dE1) , np.float32(dE2) , np.float32(dE3)]
    return dE


def gradient_descent_optimizer(image, max_iters, lr):
    # mask = np.ones(np.shape(image)) * 255
    init_theta = 0
    init_phi = 0
    init_gamma = 0

    h, w, _ = np.shape(image)
    # Feed forward
    lines = line_detection(image)
    d = np.sqrt(h ** 2 + w ** 2)
    init_f = d / 2

    for i in range(max_iters):
        # Compute gradient
        start = time.time()
        theta, phi, gamma = gradient(image, lines, init_f, init_theta, init_phi, init_gamma)

        dis = np.sqrt( theta ** 2 + phi ** 2 + gamma ** 2)
        # Normalize
        theta = theta / dis
        phi = phi / dis
        gamma = gamma / dis

        init_theta = np.float32(init_theta - lr * theta)
        init_phi = np.float32(init_phi - lr * phi)
        init_gamma = np.float32(init_gamma - 0.001 * lr * gamma)

        K = np.array([[init_f, 0, w / 2],
                      [0, init_f, h / 2],
                      [0, 0, 1]])

        K1 = np.array([[1 / init_f, 0, - w / (2 * init_f)],
                       [0, 1 / init_f, - h / (2 * init_f)],
                       [0, 0, 1]])

        R = np.array([[0, -init_gamma, init_phi],
                      [init_gamma, 0, -init_theta],
                      [-init_phi, init_theta, 0]])

        R = expm(R)

        # Add translation matrix
        dx = w / 2
        dy = h / 2
        T = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]])

        # H -1 function
        H1 = np.dot(K, np.dot(R, K1))
        H1 = np.dot(T, H1)
        cost = cost_function(h, w, init_f, lines, H1)
        print('Cost value :', cost)

        # Image after perspective transform
        image_out = cv2.warpPerspective(image.copy(), np.float32(H1), (2 * w, 2 * h))

        # mask_out = cv2.warpPerspective(mask, np.float32(H1), (2 * w, 2 * h))
        # mask_lines = lsd(mask_out[:, :, 0])

        # for i in range(mask_lines.shape[0]):
        #     pt1 = (int(mask_lines[i, 0]), int(mask_lines[i, 1]))
        #     pt2 = (int(mask_lines[i, 2]), int(mask_lines[i, 3]))
        #     cv2.line(mask_out, pt1, pt2, (0, 0, 255), int(2))

        # cv2.imwrite('mask_1.jpg', mask_out)
        # cv2.imwrite('image_out_' + str(1) + '.png', image_out)

    top_left = np.dot(H1, [0, 0, 1])
    top_right = np.dot(H1, [w, 0, 1])
    bottom_left = np.dot(H1, [0, h, 1])
    bottom_right = np.dot(H1, [w, h, 1])

    points = [top_left, top_right, bottom_left, bottom_right]
    image_crop = crop_black_border(image_out, points)

    # cv2.circle(image_out, (int(top_left[0] / top_left[2]), int(top_left[1] / top_left[2])), 20, (0, 255, 0), -1)
    # cv2.circle(image_out, (int(top_right[0] / top_right[2]), int(top_right[1] / top_right[2])), 20, (0, 255, 0), -1)
    # cv2.circle(image_out, (int(bottom_right[0] / bottom_right[2]), int(bottom_right[1] / bottom_right[2])), 20, (0, 255, 0), -1)
    # cv2.circle(image_out, (int(bottom_left[0]), int(bottom_left[1])), 20, (0, 255, 0), -1)
    # cv2.imwrite('mask_1.jpg', image_out)

    return image_crop, H1


if __name__ == '__main__':
    image = cv2.imread('/mnt/data/hades/source/sekiwa_rnd/image_enhancement/test.png')
    gradient_descent_optimizer(image, 5, 0.1)