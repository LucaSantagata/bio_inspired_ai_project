import numpy as np
from numpy import random
from settings import get_boxcar_constant, get_ga_constant
from wheel import *
from genetic_algorithm.individual import Individual
from typing import List, Union, Dict, Any
import math

from Box2D import b2Vec2
import dill as pickle
import os


def pol2b2Vec2(r_theta: tuple, rnd: int) -> b2Vec2:
    """
    Parameters:
    - r: float, vector amplitude
    - theta: float, vector angle
    Returns:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    """

    r, theta = r_theta
    x = round(r * math.cos(theta), rnd)
    y = round(r * math.sin(theta), rnd)

    return b2Vec2(x, y)


def wheels_vertices_pol_to_wheels_vertices(wheels_vertices_pol: List[tuple]):
    wheels_vertices = []
    round_length_vertices_coordinates = get_boxcar_constant("round_length_vertices_coordinates")

    for wheel_vertices_pol in wheels_vertices_pol:
        if wheel_vertices_pol:
            wheels_vertices.append(
                [pol2b2Vec2(vertex_pol, rnd=round_length_vertices_coordinates) for vertex_pol in
                 wheel_vertices_pol])
        else:
            wheels_vertices.append(None)

    return wheels_vertices


def rs_thetas_encoding(wheels: List) -> tuple:
    wheels_vertices_rs = []
    wheels_vertices_thetas = []

    for wheel in wheels:
        if wheel:
            wheel_vertices_rs, wheel_vertices_thetas = rs_thetas_to_string(wheel)

            wheels_vertices_rs.append(wheel_vertices_rs)
            wheels_vertices_thetas.append(wheel_vertices_thetas)
        else:
            wheels_vertices_rs.append(None)
            wheels_vertices_thetas.append(None)
    return wheels_vertices_rs, wheels_vertices_thetas


def rs_thetas_to_string(rs_thetas: tuple) -> tuple:
    round_length_vertices_coordinates = get_boxcar_constant("round_length_vertices_coordinates")

    rs = "".join(
        [
            (
                ("|" if i != 0 else "") + str(round(r, round_length_vertices_coordinates))
            ) for i, (r, _) in enumerate(rs_thetas)
        ]
    )

    thetas = "".join(
        [
            (
                ("|" if i != 0 else "") + str(theta)
            ) for i, (_, theta) in enumerate(rs_thetas)
        ]
    )

    return rs, thetas


def string_to_rs_thetas(str_rs: str, str_thetas: str) -> tuple:
    rs = [
        float(str_r)
        for str_r in str_rs.split("|")
    ]

    thetas = [
        int(str_theta)
        for str_theta in str_thetas.split("|")
    ]

    return rs, thetas

