import matplotlib.pyplot as plt
import numpy as np
import noise
import cv2

from PIL import Image


class TerrainGen:
    def __init__(self, width=500, height=500):
        self.width = width
        self.height = height
        self.terrain = None
        self.create_blank()

    def create_blank(self):
        self.terrain = np.zeros((self.width, self.height))

    def create_random(self):
        self.terrain = np.random.random((self.width, self.height))

    def create_new(self):
        self.create_my_map()

    def create_trigon(self):
        self.create_random()
        self.trigon_noise()
        self.normalize_terrain()

    def create_my_map(self):
        self.create_blank()
        noise = self.my_noise()
        for rindex, row in enumerate(self.terrain):
            new_row = self.moving_filter_1d(row, kernel=noise)
            self.terrain[rindex] = new_row

        self.normalize_terrain()

    def trigon_noise(self, factors=35, stepsize=0.02):
        factors = np.random.random((len(self.terrain), factors)) * 2 - 1
        for rindex, row in enumerate(self.terrain):
            for cindex, val in enumerate(row):
                val = self.get_sin_x(*factors[rindex], x0=cindex, stepsize=stepsize)
                self.terrain[rindex, cindex] = val

    def get_sin_x(self, *coeffs, x0, stepsize, ):
        out = 0
        for rank, cf in enumerate(coeffs):
            if rank == 0:
                out = cf
            else:
                out += np.sin(x0 / cf * stepsize) ** rank
        return out

    def moving_filter_1d(self, series, kernel):
        # new_series = []
        for index, value in enumerate(series):
            new_val = kernel[0]
            for prev_ind, factor in enumerate(kernel):
                index2 = index - prev_ind
                if index2 < 0:
                    break
                elif index2 == index:
                    continue
                offset = series[index2] * factor
                # print(index2, series[index2])
                new_val += offset / prev_ind

            series[index] = new_val
        # print(new_series)
        return np.array(series)

    def make_noisy(self, step_size, base=None):
        if base is None:
            base = np.random.randint(0, 9e4)
        for rindex, row in enumerate(self.terrain):
            for cindex, val in enumerate(row):
                n = noise.pnoise2(rindex / step_size, cindex / step_size, base=base)
                self.terrain[rindex, cindex] = val + n

    def normalize_terrain(self):
        self.terrain = self.terrain - self.terrain.min()
        self.terrain = self.terrain / self.terrain.max() * 255

    def my_noise(self, factor_ammount=100):
        factors = np.random.random(factor_ammount) * 2 - 1
        f_sum = np.sum(factors)
        steps = 0
        while np.abs(f_sum - 1) > 0.001 and steps < 10_000:
            new_factors = np.random.random(factor_ammount) * 2 - 1
            new_f_sum = np.sum(new_factors)

            if np.abs(new_f_sum - 1) < np.abs(f_sum - 1):
                f_sum = new_f_sum
                factors = new_factors
            # print(f_sum)
            steps += 1
        print(f"Steps taken to get factors: {steps}, sum: {f_sum}")
        return factors

    def save(self):
        # rgb = Image.fromarray(self.terrain)
        cv2.imwrite("map.png", self.terrain)
        # cv2.imwrite("map", rgb.tobytes())

    def extract(self):
        raise NotImplemented


if __name__ == "__main__":
    g1 = TerrainGen()
    g1.create_trigon()
    g1.save()
