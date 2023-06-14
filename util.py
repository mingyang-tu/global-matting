import numpy as np
import cv2
import math


def calculate_alpha(pixel, foreground, background):
    f_minus_b = foreground - background
    result = np.dot((pixel - background), f_minus_b) / (np.dot(f_minus_b, f_minus_b) + 1e-6)
    return max(min(result, 1), 0)


def color_cost(pixel, foreground, background):
    alpha = calculate_alpha(pixel, foreground, background)
    cost = pixel - (alpha * foreground + (1 - alpha) * background)
    return np.linalg.norm(cost)


def distance(p1, p2):
    d0 = p1[0] - p2[0]
    d1 = p1[1] - p2[1]
    return math.sqrt(float(d0 * d0 + d1 * d1))


def space_cost(position1, position2, nearest):
    return distance(position1, position2) / nearest


def calculate_cost(pixel, foreground, background, p_position, f_position, b_position, fore_nearest, back_nearest):
    cost_c = color_cost(pixel, foreground, background)
    cost_sf = space_cost(p_position, f_position, fore_nearest)
    cost_sb = space_cost(p_position, b_position, back_nearest)
    return cost_c + cost_sf + cost_sb


def get_borders(image, foreground, background, unknown):
    unknown = unknown.astype(np.uint8)

    row_idx, col_idx = np.mgrid[0: unknown.shape[0], 0: unknown.shape[1]]

    dilated = cv2.dilate(unknown, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)

    fore_edge = foreground & dilated
    back_edge = background & dilated

    fore_idx = [row_idx[fore_edge], col_idx[fore_edge]]
    back_idx = [row_idx[back_edge], col_idx[back_edge]]

    fore_idx_sorted = sort_borders(image, fore_idx)
    back_idx_sorted = sort_borders(image, back_idx)

    return fore_idx_sorted, back_idx_sorted


def sort_borders(image, indexs):
    values = image[indexs[0], indexs[1], :]
    bgr2gray = np.array([19/256, 183/256, 54/256], dtype=np.float64)
    intensity = np.dot(values, bgr2gray)
    sorted_idxs = np.argsort(intensity)
    return [(indexs[0][i], indexs[1][i]) for i in sorted_idxs]


def get_nearest_distance(foreground, background, unknown):
    fore_dist = _get_nearest_distance(unknown, foreground)
    back_dist = _get_nearest_distance(unknown, background)
    return fore_dist, back_dist


def _get_nearest_distance(unknown, ground):
    output = np.zeros(unknown.shape, dtype=np.float64)

    unseen = unknown.copy()
    distance = 1

    dilated = ground.astype(np.uint8)
    kernels = [
        np.ones((3, 3), dtype=np.uint8),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    ]
    while np.any(unseen):
        dilated = cv2.dilate(dilated, kernels[distance % 2], iterations=1)
        current = dilated.astype(bool) & unseen
        unseen[current] = False
        output[current] = float(distance)
        distance += 1
    output[unknown == False] = 0
    return output


def color_guided_filter(guided, target, radius, epsilon):
    ROW, COL = target.shape
    kernel = (radius * 2 + 1, radius * 2 + 1)

    mean_I = cv2.blur(guided, kernel)
    mean_p = cv2.blur(target, kernel)

    corr_I = np.zeros((ROW, COL, 9), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            pix = np.reshape(guided[i, j, :], (1, 3))
            corr_I[i, j, :] = np.ravel(np.dot(pix.T, pix))
    corr_I = cv2.blur(corr_I, kernel)

    corr_Ip = np.zeros((ROW, COL, 3), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            corr_Ip[i, j, :] = target[i, j] * guided[i, j, :]
    corr_Ip = cv2.blur(corr_Ip, kernel)

    eps_U = epsilon * np.eye(3, dtype=np.float64)

    mat_a = np.zeros((ROW, COL, 3), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mu = np.reshape(mean_I[i, j, :], (1, 3))
            sigma = np.reshape(corr_I[i, j, :], (3, 3)) - np.dot(mu.T, mu)
            mat_a[i, j, :] = np.dot(
                np.linalg.inv(sigma + eps_U),
                corr_Ip[i, j, :] - mean_I[i, j, :] * mean_p[i, j]
            )

    mat_b = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mat_b[i, j] = mean_p[i, j] - np.dot(mat_a[i, j, :], mean_I[i, j, :])

    mean_a = cv2.blur(mat_a, kernel)
    mean_b = cv2.blur(mat_b, kernel)

    mat_q = np.zeros((ROW, COL), dtype=np.float64)
    for i in range(ROW):
        for j in range(COL):
            mat_q[i, j] = np.dot(mean_a[i, j, :], guided[i, j, :]) + mean_b[i, j]

    return mat_q
