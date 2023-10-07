import numpy as np
import cv2
import os
import math
from scipy.spatial import KDTree


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

    row_idx, col_idx = np.mgrid[0 : unknown.shape[0], 0 : unknown.shape[1]]

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
    bgr2gray = np.array([19 / 256, 183 / 256, 54 / 256], dtype=np.float64)
    intensity = np.dot(values, bgr2gray)
    sorted_idxs = np.argsort(intensity)
    return [(indexs[0][i], indexs[1][i]) for i in sorted_idxs]


def get_nearest_distance(foreground, background, unknown):
    fore_dist = _get_nearest_distance(unknown, foreground)
    back_dist = _get_nearest_distance(unknown, background)
    return fore_dist, back_dist


def _get_nearest_distance(unknown, ground):
    unknown_points = np.array(unknown, dtype=np.float64)
    contour_points = np.array(ground, dtype=np.float64)
    kdtree = KDTree(contour_points)
    distances, _ = kdtree.query(unknown_points)
    return distances


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
            mat_a[i, j, :] = np.dot(np.linalg.inv(sigma + eps_U), corr_Ip[i, j, :] - mean_I[i, j, :] * mean_p[i, j])

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


class GlobalMatting:
    def __init__(self, image, trimap, iteration=10):
        self.image = image.astype(np.float64)
        self.iteration = iteration

        trimap = trimap.astype(np.float64)
        trimap[(trimap != 0) & (trimap != 255)] = 0.5
        trimap[trimap == 255] = 1.0

        self.row, self.col = trimap.shape

        f_bin = trimap == 1
        b_bin = trimap == 0
        self.u_bin = trimap == 0.5

        row_idx, col_idx = np.mgrid[0 : self.row, 0 : self.col]
        self.u_position = [(i, j) for (i, j) in zip(row_idx[self.u_bin], col_idx[self.u_bin])]

        self.f_position, self.b_position = get_borders(self.image, f_bin, b_bin, self.u_bin)
        self.f_nearest, self.b_nearest = get_nearest_distance(self.f_position, self.b_position, self.u_position)

        self.f_len = len(self.f_position)
        self.b_len = len(self.b_position)
        self.u_len = len(self.u_position)

        self.f_index = np.array(np.random.randint(0, self.f_len, size=(self.row, self.col)))
        self.b_index = np.array(np.random.randint(0, self.b_len, size=(self.row, self.col)))

        self.min_cost = np.full((self.row, self.col), np.inf)

        self.alpha = np.zeros((self.row, self.col), dtype=np.float64)
        self.alpha[f_bin] = 1.0

        self.windows = []
        curr = max(self.f_len, self.b_len)
        while curr >= 1:
            self.windows.append(curr)
            curr /= 2

    def run(self):
        for k in range(self.iteration):
            print(f"Iteration {k + 1} / {self.iteration}")
            indexs = np.arange(self.u_len)
            np.random.shuffle(indexs)
            for idx in indexs:
                self.propagation(idx)
                self.random_search(idx)

        print("Post processing...")
        post_process = color_guided_filter(self.image, self.alpha, 10, 1e-6)
        return post_process

    def propagation(self, idx):
        (i, j) = self.u_position[idx]
        neighbors = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        for di, dj in neighbors:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.row and 0 <= new_j < self.col and self.u_bin[new_i, new_j]:
                f, b = self.f_index[new_i, new_j], self.b_index[new_i, new_j]
                fi, fj = self.f_position[f]
                bi, bj = self.b_position[b]
                cost = calculate_cost(
                    self.image[i, j, :],
                    self.image[fi, fj, :],
                    self.image[bi, bj, :],
                    (i, j),
                    (fi, fj),
                    (bi, bj),
                    self.f_nearest[idx],
                    self.b_nearest[idx],
                )
                if cost < self.min_cost[i, j]:
                    self.min_cost[i, j] = cost
                    self.f_index[i, j] = f
                    self.b_index[i, j] = b
                    self.alpha[i, j] = calculate_alpha(
                        self.image[i, j, :], self.image[fi, fj, :], self.image[bi, bj, :]
                    )

    def random_search(self, idx):
        (i, j) = self.u_position[idx]
        for w in self.windows:
            f = self.f_index[i, j] + round(w * (np.random.random() * 2 - 1))
            f = min(max(0, f), self.f_len - 1)
            b = self.b_index[i, j] + round(w * (np.random.random() * 2 - 1))
            b = min(max(0, b), self.b_len - 1)
            fi, fj = self.f_position[f]
            bi, bj = self.b_position[b]
            cost = calculate_cost(
                self.image[i, j, :],
                self.image[fi, fj, :],
                self.image[bi, bj, :],
                (i, j),
                (fi, fj),
                (bi, bj),
                self.f_nearest[idx],
                self.b_nearest[idx],
            )
            if cost < self.min_cost[i, j]:
                self.min_cost[i, j] = cost
                self.f_index[i, j] = f
                self.b_index[i, j] = b
                self.alpha[i, j] = calculate_alpha(self.image[i, j, :], self.image[fi, fj, :], self.image[bi, bj, :])


if __name__ == "__main__":
    root = "test-images"

    image = cv2.imread(os.path.join(root, "input.png"))
    trimap = cv2.imread(os.path.join(root, "trimap.png"), cv2.IMREAD_GRAYSCALE)

    matting = GlobalMatting(image, trimap)
    alpha_map = matting.run()

    cv2.imshow("input", image)
    cv2.imshow("trimap", trimap)
    cv2.imshow("alpha map", alpha_map)
    # cv2.imwrite("result.png", np.clip(alpha_map*255, 0, 255).astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
