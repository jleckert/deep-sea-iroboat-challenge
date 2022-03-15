from src import geometry_helper
from math import pi


def test_vr_trigo_bijection():
    assert geometry_helper.vr_trigo_bijection(0) == 90
    assert geometry_helper.vr_trigo_bijection(90) == 0
    assert geometry_helper.vr_trigo_bijection(180) == 270
    assert geometry_helper.vr_trigo_bijection(270) == 180
    assert geometry_helper.vr_trigo_bijection(45) == 45


def test_angleFromCoordinate():
    assert int(geometry_helper.angleFromCoordinate(
        5 * pi / 180, 10 * pi / 180, 5 * pi / 180, 15 * pi / 180)) == 89

    assert int(geometry_helper.angleFromCoordinate(
        5 * pi / 180, 10 * pi / 180, 10 * pi / 180, 10 * pi / 180)) == 0

    assert int(geometry_helper.angleFromCoordinate(
        5 * pi / 180, 5 * pi / 180, 10 * pi / 180, 10 * pi / 180)) == 44

    assert int(geometry_helper.angleFromCoordinate(
        10 * pi / 180, 10 * pi / 180, 10 * pi / 180, 5 * pi / 180)) == 270


def test_filter_wp_angle():
    assert geometry_helper.filter_wp_angle(
        0, 0, [[1, 1], [2, 2]]) == [[1, 1], [2, 2]]

    assert geometry_helper.filter_wp_angle(
        0, 0, [[1, 1], [2, 2], [2, 10]]) == [[1, 1], [2, 2]]

    assert geometry_helper.filter_wp_angle(
        0, 0, [[1, 1], [2, 2], [10, 2]]) == [[1, 1], [2, 2]]

    assert geometry_helper.filter_wp_angle(
        0, 0, [[1, 1], [10, 2]]) == [[1, 1]]
