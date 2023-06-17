import numpy as np
import cv2
import os
from util import calculate_alpha, calculate_cost, get_borders, get_nearest_distance, color_guided_filter


class GlobalMatting:
    def __init__(self, image, trimap, iteration=10):
        self.image = image.astype(np.float64)
        self.iteration = iteration

        trimap = trimap.astype(np.float64)
        trimap[(trimap != 0) & (trimap != 255)] = 0.5
        trimap[trimap == 255] = 1.

        self.row, self.col = trimap.shape

        f_bin = (trimap == 1)
        b_bin = (trimap == 0)
        self.u_bin = (trimap == 0.5)

        row_idx, col_idx = np.mgrid[0: self.row, 0: self.col]
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
        self.alpha[f_bin] = 1.

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
        for (di, dj) in neighbors:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.row and 0 <= new_j < self.col and self.u_bin[new_i, new_j]:
                f, b = self.f_index[new_i, new_j], self.b_index[new_i, new_j]
                fi, fj = self.f_position[f]
                bi, bj = self.b_position[b]
                cost = calculate_cost(
                    self.image[i, j, :], self.image[fi, fj, :], self.image[bi, bj, :],
                    (i, j), (fi, fj), (bi, bj),
                    self.f_nearest[idx], self.b_nearest[idx]
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
                self.image[i, j, :], self.image[fi, fj, :], self.image[bi, bj, :],
                (i, j), (fi, fj), (bi, bj),
                self.f_nearest[idx], self.b_nearest[idx]
            )
            if cost < self.min_cost[i, j]:
                self.min_cost[i, j] = cost
                self.f_index[i, j] = f
                self.b_index[i, j] = b
                self.alpha[i, j] = calculate_alpha(
                    self.image[i, j, :], self.image[fi, fj, :], self.image[bi, bj, :]
                )


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
