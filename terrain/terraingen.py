# import matplotlib.pyplot as plt
import numpy as np
import noise
import time
import cv2

# GAUSSIAN_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
global function_stats
function_stats = dict()


def timeit(fun):
    def wrapper(*args, **kwargs):
        global function_stats
        name = fun.__name__
        time0 = time.time()
        value = fun(*args, **kwargs)
        timend = time.time()

        dur_s = (timend - time0)
        records = function_stats.get(name, [])
        records.append(dur_s)
        function_stats[name] = records
        # print(f"{fun.__name__:<20} took: {dur_s:>05.2f} s")
        return value

    return wrapper


@timeit
def GaussianKernel(size=11):
    arr = np.zeros((size, size))
    padding = size // 2
    for ind, row in enumerate(arr):
        for cind, col in enumerate(row):
            distance = np.abs(padding - ind) + np.abs(padding - cind)
            distance = distance

            arr[ind, cind] = (2 * size - distance)
    return arr


# @timeit
# def DilatationKernel(size=11):
#     arr = np.zeros((size, size))
#     padding = size // 2
#     for ind, row in enumerate(arr):
#         for cind, col in enumerate(row):
#             distance = np.abs(padding - ind) + np.abs(padding - cind)
#             distance = distance
#
#             arr[ind, cind] = (2 * size - distance)
#     return arr


class TerrainGen:
    def __init__(self, width=1500, height=1500):
        self.width = width
        self.height = height

        self.bgr = None
        self.components = dict()
        self.terrain = self.create_blank()

    def create_blank(self, value=0.0, chanels=1):
        blank = np.zeros((self.width, self.height, chanels)) + value
        return blank

    @timeit
    def create_random(self):
        terrain = np.random.random((self.width, self.height, 1))
        return terrain

    @timeit
    def create_trigon(self, seed=None, point=None):
        # if seed is None:
        #     seed = np.random.randint(0, 1e3)

        # print(f"Seed: {seed}, Offset: {offsetx}, {offsety}")

        self.terrain = self.create_blank()
        N = 8
        for x in range(1, 6):
            pointx, pointy = np.random.random(2) * 5 + 1
            land0 = self.get_trigon_noise(factors=N, x0=pointx, y0=pointy, stepsize=1e-3, amplitude=10 / x)
            land1 = self.get_trigon_noise(factors=N, x0=pointx, y0=pointy, stepsize=0.5e-3, amplitude=3 / x)
            self.terrain += land0 + land1
        # moutains = self.get_trigon_noise(factors=3, start_point=point, stepsize=1e-3, amplitude=2)
        # rocks = self.get_trigon_noise(factors=3, start_point=point, stepsize=1e-3, amplitude=2)
        # river = self.get_trigon_noise(factors=3, start_point=point, stepsize=1e-3, amplitude=2)

        self.terrain = self.normalize_terrain(self.terrain)
        self.bgr = self.get_color_map(self.terrain)

    # @timeit
    # def create_my_map(self):
    #     self.terrain = self.create_blank()
    #     noise = self.my_noise()
    #     for rindex, row in enumerate(self.terrain):
    #         new_row = self.moving_filter_1d(row, kernel=noise)
    #         self.terrain[rindex] = new_row
    #
    #     self.terrain = self.normalize_terrain()

    @timeit
    def get_trigon_noise(self, factors=2, stepsize=1e-3, x0=1e6, y0=1e6, amplitude=1.0):
        factors = np.random.random((2, factors)) * 2
        xval = [self.get_sin_x(*factors[0], x0=x0 + i * stepsize) for i in range(self.width)]
        yval = [self.get_sin_x(*factors[1], x0=y0 + i * stepsize) for i in range(self.height)]

        XX, YY = np.meshgrid(xval, yval)
        ZZ = XX * YY
        ZZ = ZZ.reshape(*ZZ.shape, -1) * amplitude
        return ZZ

    @timeit
    def create_perlin(self, step_size=0.1, seed=None, offsetx=0, offsety=0):
        if seed is None:
            seed = np.random.randint(0, 1e3)

        if offsetx == 0 and offsety == 0:
            offsetx, offsety = np.random.randint(0, 1e4, 2)
        print(f"Seed: {seed}, Offset: {offsetx}, {offsety}")

        mountain = self.get_perlin_noise(0.2, step_size, seed, offsetx, offsety) * 255
        rocks = self.get_perlin_noise(0.03, step_size * 4, seed + 1, offsetx, offsety) * 255
        river = self.get_perlin_noise(0.8, step_size / 2, seed + 2, offsetx, offsety) * 255

        self.components = dict(mountain=mountain, rocks=rocks, river=river)
        terrain = self.create_blank(0.5) + mountain + rocks + river

        self.terrain = self.normalize_terrain(terrain)
        self.bgr = self.get_color_map(self.terrain)

        new_terrain = self.work_on_terrain(self.bgr, self.terrain)
        bgr_new = self.get_color_map(new_terrain)

        debug = np.concatenate([new_terrain, new_terrain, new_terrain], -1)
        self.components['debug'] = debug
        self.components['map_better'] = bgr_new

    @timeit
    def work_on_terrain(self, bgr_map, terrain) -> 'List 2d Terrain':
        """

        Args:
            bgr_map:
            terrain:

        Returns:
        2D np.array
        """
        red_mask = bgr_map[:, :, 2] <= 10
        green_mask = bgr_map[:, :, 1] <= 10
        blue_mask = bgr_map[:, :, 0] >= 10

        mask_bool = np.logical_and(blue_mask, np.logical_or(red_mask, green_mask))
        mask = mask_bool * 255

        mask = np.array(mask, dtype=np.float).reshape(*mask.shape, 1)
        new_terrain = self.apply_filter(terrain, GaussianKernel(33), mask)

        dil_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (25, 25))
        mask_arr = cv2.dilate(mask, kernel=dil_kernel, iterations=1)
        mask_arr = mask_arr.reshape(*mask_arr.shape, 1)

        new_terrain = self.apply_filter(new_terrain, GaussianKernel(11), mask_arr)
        return new_terrain

    @timeit
    def apply_filter(self, terrain, kernel, mask=None):
        output = terrain.copy()
        application = self.create_blank(chanels=3)
        application[:, :, 0] = mask[:, :, 0]

        if mask is None:
            mask = self.create_blank(0)

        if not len(kernel) % 2:
            print(f"Dimension row is even: {len(kernel)}")
            return output

        if not len(kernel[0]) % 2:
            print(f"Dimension of columns is even: {len(kernel[0])}")
            return output

        factor = kernel.sum()
        padding_vert = kernel.shape[0] // 2
        padding_hor = kernel.shape[1] // 2

        for rindex, (row, mask_row) in enumerate(zip(
                terrain[padding_vert:-padding_vert], mask[padding_vert:]), padding_vert):

            row_indexes = slice(rindex - padding_vert, rindex + padding_vert + 1)
            for cindex, (height, msk) in enumerate(zip(
                    row[padding_hor:-padding_hor], mask_row[padding_hor:]), padding_hor):
                col_indexes = slice(cindex - padding_hor, cindex + padding_hor + 1)
                if msk < 1:
                    continue
                else:
                    scrap = terrain[row_indexes, col_indexes, 0]
                    application[rindex, cindex, 1] = 255
                    val = (scrap * kernel).sum() / factor
                output[rindex, cindex, 0] = val

        self.components['application'] = application
        return output

    # @timeit
    # def moving_filter_1d(self, series, kernel):
    #     # new_series = []
    #     for index, value in enumerate(series):
    #         new_val = kernel[0]
    #         for prev_ind, factor in enumerate(kernel):
    #             index2 = index - prev_ind
    #             if index2 < 0:
    #                 break
    #             elif index2 == index:
    #                 continue
    #             offset = series[index2] * factor
    #             # print(index2, series[index2])
    #             new_val += offset / prev_ind
    #
    #         series[index] = new_val
    #     # print(new_series)
    #     return np.array(series)

    @timeit
    def get_sin_x(self, *coeffs, x0, ):
        out = 0
        for rank, cf in enumerate(coeffs):
            if rank == 0:
                out = cf
            else:
                out += np.sin((x0) ** cf) / rank
        if out > 1:
            out = 1
        return out

    @timeit
    def get_color_map(self, terrain, water_volume=0.2, grass_volume=0.5, rock_volume=0.2):
        bgr_map = self.create_blank(chanels=3)

        rav = terrain.copy().ravel()
        rav.sort()
        target_water = int(len(rav) * water_volume)
        target_grass = int(target_water + len(rav) * grass_volume)
        target_rock = int(target_grass + len(rav) * rock_volume)
        water_height = rav[target_water]
        grass_height = rav[target_grass]
        rock_height = rav[target_rock]

        for rindex, row in enumerate(terrain):
            for cindex, val in enumerate(row):
                if val <= water_height:
                    bgr_map[rindex, cindex, 0] = 255
                elif val <= grass_height:
                    bgr_map[rindex, cindex, 1] = 255
                elif val <= rock_height:
                    bgr_map[rindex, cindex, :] = 125
                else:
                    bgr_map[rindex, cindex, :] = 255

        bgr_map = self.normalize_terrain(bgr_map)
        return bgr_map

    @timeit
    def get_perlin_noise(self, amplitude, step_size, seed, offsetx, offsety):
        noise_map = self.create_blank()
        for rindex, row in enumerate(self.terrain):
            for cindex, val in enumerate(row):
                n = noise.pnoise2((rindex + offsetx) * step_size,
                                  (cindex + offsety) * step_size,
                                  base=seed)
                noise_map[rindex, cindex] = n * amplitude
        return noise_map

    @timeit
    def normalize_terrain(self, terrain=None, minimal_val=0, maximal_val=255):
        terrain = terrain - terrain.min()

        if terrain.min() == terrain.max():
            print(f"Map is flat")
            return terrain
        terrain = terrain / terrain.max() * (maximal_val - minimal_val) + minimal_val
        return terrain

    @staticmethod
    def my_noise(factor_ammount=100):
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

    @timeit
    def blur_terrain(self, terrain):
        terrain = cv2.GaussianBlur(terrain, (15, 15), 10)
        return terrain

    @timeit
    def save(self):
        terrain_bgr = np.concatenate([self.terrain] * 3, axis=-1)

        if self.bgr is not None:
            cv2.imwrite("map_bgr.png", self.bgr)
            stacked = np.concatenate([self.bgr, terrain_bgr], axis=1)
        else:
            stacked = terrain_bgr

        if 'map_better' in self.components and 'debug' in self.components:
            layer = np.concatenate([self.components['map_better'], self.components['debug']], axis=1)
            stacked = np.concatenate([stacked, layer], axis=0)

        elif 'debug' in self.components:
            stacked = np.concatenate([stacked, self.components['debug']], axis=1)

        cv2.imwrite("stacked.png", stacked)
        cv2.imwrite("map.png", self.terrain)

        for name, im in self.components.items():
            cv2.imwrite(f"{name}.png", im)

    def extract(self):
        raise NotImplemented


if __name__ == "__main__":
    t0 = time.time()
    g1 = TerrainGen(300, 300)
    g1.create_perlin(step_size=0.008)
    # g1.create_trigon()
    g1.save()
    tend = time.time()

    for key, records in function_stats.items():
        print(f"{key:<25} average: {len(records):>5}x {np.mean(records):>4.4f}s \t"
              f"sum: {np.sum(records):>4.2f}s")

    print(f"Whole program took: {tend - t0:>4.1f} s")
