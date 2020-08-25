# import matplotlib.pyplot as plt
import numpy as np
import noise
import cv2


# from PIL import Image, ImageFilter


class TerrainGen:
    def __init__(self, width=1500, height=1500):
        self.width = width
        self.height = height
        self.terrain = None
        self.red = None
        self.green = None
        self.blue = None
        self.terrain = self.create_blank()

    def create_blank(self, chanels=1):
        blank = np.zeros((self.width, self.height, chanels))
        return blank

    def create_random(self):
        self.terrain = np.random.random((self.width, self.height))

    def create_trigon(self):
        self.create_random()
        for x in range(1, 11):
            self.add_trigon_noise(factors=x + 1,
                                  stepsize=1e-3, amplitude=10 / x)
        self.blur_terrain()
        self.terrain = self.normalize_terrain()

    def create_my_map(self):
        self.terrain = self.create_blank()
        noise = self.my_noise()
        for rindex, row in enumerate(self.terrain):
            new_row = self.moving_filter_1d(row, kernel=noise)
            self.terrain[rindex] = new_row

        self.terrain = self.normalize_terrain()

    def add_trigon_noise(self, factors=2, stepsize=1e-3, start_point=1e6, amplitude=1.0):
        factors = np.random.random((2, factors)) * 2 - 1
        xval = [self.get_sin_x(*factors[0], x0=start_point + i, stepsize=stepsize) for i in range(self.width)]
        yval = [self.get_sin_x(*factors[1], x0=start_point + i, stepsize=stepsize) for i in range(self.height)]

        XX, YY = np.meshgrid(xval, yval)
        ZZ = XX * YY

        self.terrain += ZZ * amplitude

    def get_sin_x(self, *coeffs, x0, stepsize, ):
        out = 0
        for rank, cf in enumerate(coeffs):
            if rank == 0:
                out = cf
            else:
                out += np.sin((x0 * stepsize) ** cf) / rank
        if out > 1:
            out = 1
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

    def create_perlin(self, step_size=0.1, seed=None, offsetx=0, offsety=0):
        if seed is None:
            seed = np.random.randint(0, 1e3)

        if offsetx == 0 and offsety == 0:
            offsetx, offsety = np.random.randint(0, 1e4, 2)
            print(offsetx, offsety)

        mountain = self.get_perlin_noise(0.4, step_size, seed, offsetx, offsety) * 255
        rocks = self.get_perlin_noise(0.03, step_size * 8, seed + 1, offsetx, offsety) * 255
        river = self.get_perlin_noise(0.8, step_size, seed + 2, offsetx, offsety) * 255

        self.red = mountain
        self.green = rocks
        self.blue = river
        self.terrain = self.create_blank() + 0.5 + mountain + rocks - river

    def get_perlin_noise(self, amplitude, step_size, seed, offsetx, offsety):
        noise_map = self.create_blank()
        for rindex, row in enumerate(self.terrain):
            for cindex, val in enumerate(row):
                n = noise.pnoise2((rindex + offsetx) * step_size,
                                  (cindex + offsety) * step_size,
                                  base=seed)
                noise_map[rindex, cindex] = n * amplitude
        return noise_map

    def normalize_terrain(self, terrain=None, minimal_val=0, maximal_val=255):
        terrain = terrain - terrain.min()

        if terrain.min() == terrain.max():
            print(f"Map is flat")
            return terrain
        terrain = terrain / terrain.max() * (maximal_val - minimal_val) + minimal_val
        return terrain

    def normalize_terrain_3d(self):
        self.red = self.normalize_terrain(self.red)
        self.green = self.normalize_terrain(self.green)
        self.blue = self.normalize_terrain(self.blue)

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
        # self.normalize_terrain_3d()
        im = np.concatenate([self.red, self.green, self.blue], axis=-1)
        rgb = self.normalize_terrain(im)
        self.terrain = self.normalize_terrain(self.terrain)

        rgb = np.array(rgb, dtype=np.uint8)
        cv2.imwrite("map_rgb.png", rgb)
        cv2.imwrite("map.png", self.terrain)
        cv2.imwrite("map_red.png", self.red)
        cv2.imwrite("map_blue.png", self.blue)
        cv2.imwrite("map_green.png", self.green)

    def extract(self):
        raise NotImplemented


if __name__ == "__main__":
    g1 = TerrainGen(300, 300)
    g1.create_perlin(step_size=0.003)
    g1.save()
