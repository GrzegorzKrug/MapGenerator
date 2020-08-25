import matplotlib.pyplot as plt
import numpy as np
import noise
import cv2

from PIL import Image, ImageFilter


class TerrainGen:
    def __init__(self, width=1500, height=1500):
        self.width = width
        self.height = height
        self.terrain = None
        self.create_blank()

    def create_blank(self):
        self.terrain = np.zeros((self.width, self.height))

    def create_random(self):
        self.terrain = np.random.random((self.width, self.height))

    def create_trigon(self):
        self.create_random()
        for x in range(1, 10):
            self.add_trigon_noise(factors=x + 3, stepsize=x / 500)
        self.blur_terrain()
        self.normalize_terrain()

    def create_my_map(self):
        self.create_blank()
        noise = self.my_noise()
        for rindex, row in enumerate(self.terrain):
            new_row = self.moving_filter_1d(row, kernel=noise)
            self.terrain[rindex] = new_row

        self.normalize_terrain()

    def add_trigon_noise(self, factors=8, stepsize=1e-3, start_point=1e6):
        factors = np.random.random((2, factors)) * 2 - 1
        xval = [self.get_sin_x(*factors[0], x0=start_point + i, stepsize=stepsize) for i in range(self.width)]
        yval = [self.get_sin_x(*factors[1], x0=start_point + i, stepsize=stepsize) for i in range(self.height)]

        XX, YY = np.meshgrid(xval, yval)
        ZZ = XX * YY

        self.terrain += ZZ

    def get_sin_x(self, *coeffs, x0, stepsize, ):
        out = 0
        for rank, cf in enumerate(coeffs):
            if rank == 0:
                out = cf
            else:
                out += np.sin((x0 * stepsize) ** cf)
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

    def normalize_terrain(self, minimal_val=0, maximal_val=255):
        self.terrain = self.terrain - self.terrain.min()
        self.terrain = self.terrain / self.terrain.max() * (maximal_val - minimal_val) + minimal_val

    def normalize_terrain_2(self):
        self.terrain = self.terrain - self.terrain.min()
        self.terrain = self.terrain * 255

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

    def blur_terrain(self):
        self.terrain = cv2.GaussianBlur(self.terrain, (15, 15), 10)

    def save(self):
        # im = np.stack([red, green, blue], axis=-1)
        im = np.array(self.terrain, dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("map_rgb.png", im)

    def extract(self):
        raise NotImplemented


if __name__ == "__main__":
    g1 = TerrainGen(300, 300)
    g1.create_trigon()
    g1.save()
