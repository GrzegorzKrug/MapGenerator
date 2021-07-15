from terraingen import TerrainGen

import pytest
import time


def run_in_time(sec_req=10, repeat=1, assert_overtime=False):
    def decorator(fun):
        def wrapper(*a, **kw):

            for n in range(repeat):
                t0 = time.time()
                fun(*a, **kw)
                tend = time.time()
                dur = tend - t0

                if not assert_overtime:
                    if dur > sec_req:
                        raise TimeoutError("Generation was too long")

                elif dur < sec_req:
                    raise ValueError("Generation was too fast")

        return wrapper

    return decorator


def test_1_run_generator():
    TerrainGen()


@run_in_time(0.1, assert_overtime=True)
# @pytest.mark.timeout(0.3)
def test_2_checking_timing_function():
    time.sleep(1)


@run_in_time(0.3, assert_overtime=False)
# @pytest.mark.timeout(0.3)
def test_2_checking_timing_function():
    time.sleep(0.1)


def test_3_MapGen_Seed_save_output():
    m1 = TerrainGen(seed="4")
    m2 = TerrainGen(seed="4")
    assert m1 == m2


def test_4_MapGen_Symetric_x():
    raise NotImplemented


def test_5_MapGen_Symetric_Radial():
    raise NotImplemented


def test_6_MapGen_Markers():
    raise NotImplemented


def test_7_MapGen_Enemy_Armies():
    raise NotImplemented


def test_8_():
    raise NotImplemented


def test_9_():
    raise NotImplemented


def test_10_():
    raise NotImplemented


def test_11_():
    raise NotImplemented


def test_12_():
    raise NotImplemented


def test_13_():
    raise NotImplemented


def test_14_():
    raise NotImplemented


def test_15_():
    raise NotImplemented


def test_16_():
    raise NotImplemented


def test_17_():
    raise NotImplemented


def test_18_():
    raise NotImplemented


def test_19_():
    raise NotImplemented


def test_20_():
    raise NotImplemented


def test_21_():
    raise NotImplemented


def test_22_():
    raise NotImplemented


def test_23_():
    raise NotImplemented


def test_24_():
    raise NotImplemented


def test_25_():
    raise NotImplemented
