# import matplotlib.pyplot as plt
import numba
import numpy as np
import noise
import time
import glob
import cv2
import sys
import os

from scipy.fft import fft2, fftshift
from sklearn.cluster import KMeans

# GAUSSIAN_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
global function_stats


# function_stats = dict()

class Timer:
    times = []
    groups = {}
    t0 = time.time()

    @classmethod
    def timeit(cls, fun):
        def wrapper(*args, **kwargs):
            global function_stats
            name = fun.__name__
            time0 = time.time()
            value = fun(*args, **kwargs)
            timend = time.time()

            dur_s = (timend - time0)

            "Excecution order"
            cls.times.append((name, dur_s))

            "Grouping times"
            if name in cls.groups:
                cls.groups[name].append(dur_s)
            else:
                cls.groups[name] = [dur_s]

            return value

        return wrapper

    @staticmethod
    def get_prec(precision=4):
        fmt = f">{precision}"
        float_fmt = f">4.{precision}f"
        return fmt, float_fmt

    @classmethod
    def print_average(cls, precision=4):
        fmt, float_fmt = cls.get_prec(precision)

        for key, records in cls.groups.items():
            print(f"{key:<25} average: {len(records):{fmt}}x {np.mean(records):{float_fmt}} s \t"
                  f"sum: {np.sum(records):{float_fmt}} s")

    @classmethod
    def print_sum(cls, precision=2):
        fmt, float_fmt = cls.get_prec(precision)
        # ttimes = [t for n, t in cls.times]
        exec_time = time.time() - cls.t0
        print(f"Whole program took: {exec_time:{float_fmt}} s")

    @classmethod
    def clear(cls):
        cls.groups = {}
        cls.times = []
        cls.to = time.time()


@Timer.timeit
def GaussianKernel(size=11):
    arr = np.zeros((size, size))
    padding = size // 2
    for ind, row in enumerate(arr):
        for cind, col in enumerate(row):
            distance = np.abs(padding - ind) + np.abs(padding - cind)
            distance = distance

            arr[ind, cind] = (2 * size - distance)
    return arr


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
    def __init__(self, width=1500, height=1500, seed=None):
        self.width = width
        self.height = height

        self.bgr = None
        # self.components = dict()
        self.terrain = self.create_blank(h=height, w=width)

    @staticmethod
    def create_blank(val=0, h=None, w=None, channels=1):
        blank = np.zeros((h, w, channels)) + val
        return blank

    @Timer.timeit
    def create_random(self):
        terrain = np.random.random((self.width, self.height, 1))
        return terrain

    @Timer.timeit
    def create_trigon(self, seed=None, point=None):
        """Create Trigon Pattern"""
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

        self.terrain = self.normalize_terrain(self.terrain)
        self.bgr = self.get_color_map(self.terrain)

    @Timer.timeit
    def get_trigon_noise(self, factors=2, stepsize=1e-3, x0=1e6, y0=1e6, amplitude=1.0):
        factors = np.random.random((2, factors)) * 2
        xval = [self.get_sin_x(*factors[0], x0=x0 + i * stepsize) for i in range(self.width)]
        yval = [self.get_sin_x(*factors[1], x0=y0 + i * stepsize) for i in range(self.height)]

        XX, YY = np.meshgrid(xval, yval)
        ZZ = XX * YY
        ZZ = ZZ.reshape(*ZZ.shape, -1) * amplitude
        return ZZ

    # @Timer.timeit
    # def Old_create_perlin_map(self, step_size=0.001, seed=None,
    #                           offsetx=None, offsety=None, faster=True):
    #     if seed is None:
    #         seed = np.random.randint(0, 1e3)
    #     else:
    #         seed = int(seed)
    #
    #     if offsetx == 0 and offsety == 0:
    #         np.random.seed(seed)
    #         offsetx, offsety = np.random.randint(0, 1e4, 2)
    #
    #     print(f"Seed: {seed}, Offset: {offsetx}, {offsety}")
    #
    #     mountain = self.get_perlin_noise(0.2, step_size, seed, offsetx, offsety) * 255
    #     rocks = self.get_perlin_noise(0.03, step_size * 4, seed + 1, offsetx, offsety) * 255
    #     river = self.get_perlin_noise(0.8, step_size / 2, seed + 2, offsetx, offsety) * 255
    #
    #     self.components = dict(mountain=mountain, rocks=rocks, river=river)
    #     terrain = self.create_blank(0.5) + mountain + rocks + river
    #
    #     self.terrain = self.normalize_terrain(terrain)
    #     self.bgr = self.get_color_map(self.terrain)
    #
    #     new_terrain = self.work_on_terrain(self.bgr, self.terrain)
    #     bgr_new = self.get_color_map(new_terrain)
    #
    #     # debug = np.concatenate([new_terrain, new_terrain, new_terrain], -1)
    #     # self.components['debug'] = debug
    #     self.components['map_better'] = bgr_new

    @Timer.timeit
    def generate_perlin_map(self, step_size=0.01, seed=None, offsetx=None, offsety=None, faster: float = 20):
        """

        Args:
            step_size: Keep between 0.01 ; 0,004
                Lower values have artifacts at bigger zoom, but usefull for simple shapes
            seed:
            offsetx:
            offsety:
            faster: >1 Faster
                : 0<1 Upscaling

        Returns:

        """
        if seed is None:
            seed = np.random.randint(0, 1e4)
        else:
            seed = int(seed)

        h, w = self.height, self.width

        print(f"Seed: {seed:>6}, Offset: {offsetx:>7}, {offsety:>7}")

        # n = self.get_perlin_noise(step_size, seed=seed,
        #                           offsetx=offsetx, offsety=offsety, height=h, width=w,
        #                           faster=faster,
        #                           )
        #
        # self.terrain = self.normalize_terrain(n)
        # self.bgr = self.get_color_map(self.terrain)

    # @numba.njit()
    @staticmethod
    @Timer.timeit
    def get_perlin_noise(step_size, seed, offsetx=None, offsety=None, height=None, width=None, faster=None):
        if faster is not None:
            fs = float(faster)
            h, w = np.array((height, width), dtype=int) / fs
            h = int(round(h))
            w = int(round(w))
            stp = step_size * fs
        else:
            h = height
            w = width
            stp = step_size

        if type(seed) is str:
            seed = sum([ord(l) for l in seed])

        np.random.seed(seed)
        if offsetx is None or offsety is None:
            offsetx, offsety = np.random.randint(-100, 100, 2)

        noise_map = np.zeros((h, w, 1))
        for rindex in range(h):
            for cindex in range(w):
                n = noise.pnoise2(
                        rindex * stp + offsetx,
                        cindex * stp + offsety,
                        base=seed,
                )
                noise_map[rindex, cindex] = n  # * amplitude

        if faster is not None:
            noise_map = cv2.resize(noise_map, (width, height))
            noise_map = noise_map.reshape((height, width, -1))
            # noise_map = noise_map[:, :, np.newaxis]
        return noise_map

    def make_tiles(self, N, stp, seed, h, w, offx=None, offy=None, faster=None):
        if type(seed) is str:
            seed = sum([ord(l) for l in seed])
        tiles = [self.get_perlin_noise(stp, seed + n, offx, offy, h, w, faster=faster) for n in range(N)]
        tiles = [TerrainGen.normalize_terrain(tile) for tile in tiles]
        return tiles

    @Timer.timeit
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

    @Timer.timeit
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

    # def moving_filter_1d(self, series, kernel):
    #     # new_series = []
    #     for index, value in enumerate(series):
    #         new_val = kernel[0]
    #         for prev_ind, factor in enumerate(kernel):
    #             index = index - prev_ind
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

    @Timer.timeit
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

    # @Timer.timeit
    @classmethod
    def get_color_map(cls, terrain, water_volume=0.2, grass_volume=0.5, rock_volume=0.2):
        h, w, *c = terrain.shape
        bgr_map = cls.create_blank(h=h, w=w, channels=3)

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

        bgr_map = cls.normalize_terrain(bgr_map)
        return bgr_map

    @staticmethod
    @Timer.timeit
    def normalize_terrain(terrain=None, minimal_val=0, maximal_val=255):
        """Normalize matric to min/max_val"""
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

    @Timer.timeit
    def blur_terrain(self, terrain):
        terrain = cv2.GaussianBlur(terrain, (15, 15), 10)
        return terrain

    @Timer.timeit
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


def dynamic_tile_merge(t1, t2, axis=0, upscale=True, smooth_length=0.1):
    h, w, *c = t1.shape
    assert t1.shape == t2.shape, "Function works only on same tile sizes"

    if upscale:
        if axis == 0:
            out_width = w
            smooth_pixels = np.round(h * smooth_length).astype(int)
            out_height = h - smooth_pixels

            upscaled = np.array((w, h), dtype=int) * (1, 1 + smooth_length)
            upscaled = upscaled.astype(int)
            t1 = cv2.resize(t1, upscaled)
            t2 = cv2.resize(t2, upscaled)

        else:
            out_height = h
            smooth_pixels = np.round(w * smooth_length).astype(int)
            out_width = w - smooth_pixels

            upscaled = np.array((w, h), dtype=int) * (1 + smooth_length, 1)
            t1 = cv2.resize(t1, upscaled)
            t2 = cv2.resize(t2, upscaled)
        smooth_pixels *= 2

    else:
        if axis == 0:
            smooth_pixels = np.round(h * smooth_length).astype(int)
            out_height = h - smooth_pixels
            out_width = w
        else:
            smooth_pixels = np.round(w * smooth_length).astype(int)
            out_height = h
            out_width = w - smooth_pixels

        t1 = t1.reshape((h, w))
        t2 = t2.reshape((h, w))

    if axis == 0:
        # print(f"out_h:{out_height}, smooth:{smooth_pixels}, t1:{t1.shape}")
        top = t1[:out_height, :]
        bottom = t2[smooth_pixels:, :]

        roi1 = t1[out_height:, :]
        roi2 = t2[:smooth_pixels, :]

        # normalizer = smooth_pixels - 1
        gradient = np.ogrid[:smooth_pixels]
        one = np.ones((w, 1))

        bottom_mask = (gradient * one).T ** 2
        top_mask = np.flipud(bottom_mask)
        mask_sum = top_mask + bottom_mask
        normalizer = mask_sum[:, 0].reshape(-1, 1)
        # mask_sum = mask_sum / normalizer
        # print(f"Mask:\n{(mask_sum)}")
        roi1 = roi1 * top_mask / normalizer
        roi2 = roi2 * bottom_mask / normalizer
        roi_smooth = roi1 + roi2

        assert top.shape == bottom.shape, f"Shapes does not match: {top.shape}!={bottom.shape}"
        assert roi1.shape == roi2.shape, f"Shapes does not match: {roi1.shape}!={roi2.shape}"
        # print(f"Shapes: top:{top.shape}, smooth:{roi_smooth.shape}, bottom: {bottom.shape}")
        out = np.vstack([top, roi_smooth, bottom])
        return out

    return t1


def stack_tiles(tiles, h, w):
    t1 = tiles[0]
    t2 = tiles[1]

    # tl = np.vstack([t1, t2])
    tl = dynamic_tile_merge(t1, t2, upscale=True, smooth_length=0.1)
    tl = TerrainGen.normalize_terrain(tl)
    return tl


def flaterizer(tl):
    flat_margin = 10
    _X = np.ones_like(tl)
    # X = np.argwhere(_X)
    X = tl.ravel().reshape(-1, 1)
    # vals = tl.ravel()
    km = KMeans(2)
    km.fit(X)

    centers = km.cluster_centers_[0]

    # for cent in centers:
    #     mask = np.absolute(tl - cent) <= flat_margin
    #     tl[mask] = cent

    return tl


class MapFixer:
    keys = {
            "none": -1,
            "x": -2,
            "xy": -3,
            "y": -4,
            "radial": -5,
            "radial_3": -6,
    }
    rev_keys = {val: key for val, key in keys.items()}

    def __init__(self, mode, symmetry=None, flip_x=None, flip_y=None):
        pass


def clear_png():
    fils = glob.glob("*.png")
    for fil in fils:
        assert fil.endswith(".png")
        os.remove(fil)
        print(f"Removed: {fil}")


def export_stack(n=2, gap=3):
    h, w = 100, 300
    g1 = TerrainGen(w, h)
    mat = np.zeros(
            (h * n + gap * (n - 1),
             w * n + gap * (n - 1),
             3)
    )
    N = n * n
    tiles = g1.make_tiles(N, 0.02, 10, h=h, w=w, faster=3)
    ret = tiles.copy()
    for r_ind in range(n):
        for c_ind in range(n):
            out = tiles[r_ind * n + c_ind]

            gap_r = gap * r_ind
            gap_c = gap * c_ind
            mat[
            h * r_ind + gap_r:h * r_ind + h + gap_r,
            w * c_ind + gap_c:w * c_ind + w + gap_c,
            ] = out
    cv2.imwrite("tiles.png", mat)
    return ret


if __name__ == "__main__":

    tiles = export_stack()
    stk1 = stack_tiles(tiles[:2], 2, 1)
    stk2 = stack_tiles(tiles[2:], 2, 1)
    stk = stack_tiles([stk1, stk2], 2, 1)
    cv2.imwrite("stacked.png", stk)
    stk_col = TerrainGen.get_color_map(stk)
    cv2.imwrite("stacked_color.png", stk_col)

    Timer.print_average()
    Timer.print_sum()
    Timer.clear()
