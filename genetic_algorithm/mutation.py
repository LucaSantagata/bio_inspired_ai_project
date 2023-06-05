import random
import sys
import numpy as np
from typing import List, Union, Optional
from .individual import Individual
from settings import get_boxcar_constant, get_ga_constant
from encode_decode_chromosome_vertices import pol2b2Vec2, wheels_vertices_pol_to_wheels_vertices, rs_thetas_encoding, rs_thetas_to_string, string_to_rs_thetas


def mutate_vertex_old(
        r: float,
        theta: int,
        prob_mutation: float, 
        max_num_wheels_vertices: int,
        min_wheel_vertices_radius: float,
        max_wheel_vertices_radius: float
) -> tuple:
    if np.random.random() <= prob_mutation:
        r += (
                (np.random.random() * (max_wheel_vertices_radius - min_wheel_vertices_radius)) + min_wheel_vertices_radius
             ) / (max_wheel_vertices_radius - min_wheel_vertices_radius)

    if np.random.random() <= prob_mutation:
        theta += np.random.randint(low=-(360/max_num_wheels_vertices), high=(360/max_num_wheels_vertices))

    return r, theta


def mutate_vertex(
    add_remove: int,
    radius: float,
    density: float,
    rs_thetas: tuple,
    max_num_wheels_vertices: int,
    min_wheel_vertices_radius: float,
    max_wheel_vertices_radius: float,
    round_length_vertices_coordinates: int
) -> np.ndarray:
    rs, thetas = rs_thetas

    len_rs = len(rs) if rs is not None else 0

    # print("add_remove:", add_remove, "len_rs:", len_rs, "radius:", radius, "density:", density, "rs:", rs, "thetas:", thetas)

    if add_remove != 0:
        if add_remove == 1:
            num_vertices = len_rs + add_remove if len_rs >= 2 else 3

            rs = np.append(
                rs,
                [
                    round(
                        ((random.random() * (max_wheel_vertices_radius - min_wheel_vertices_radius)) + min_wheel_vertices_radius),
                        round_length_vertices_coordinates
                    ) for _ in range(num_vertices - len_rs)
                ]
            )[0 if rs is not None else 1:]
        elif add_remove == -1:
            num_vertices = len_rs + add_remove if len_rs > 3 else 1

            rs = rs[:num_vertices] if rs is not None else None

        thetas = np.arange(0, 360, 360 / num_vertices, dtype=int)

        if rs is not None:
            if num_vertices < 3:                                            # remove wheel
                # print("remove wheel")
                to_return = np.array([0, 0, None, None], dtype=object)  # radius, density, rs, thetas
            elif num_vertices > max_num_wheels_vertices:                    # polygon to circle wheel
                # print("polygon to circle wheel")
                to_return = np.array([radius, density, None, None], dtype=object)  # radius, density, rs, thetas
            else:                                                           # new polygon wheel
                # print("new polygon wheel")
                rs_thetas = [(r, theta) for r, theta in zip(rs, thetas)]
                to_return = np.array([radius, density, *rs_thetas_to_string(rs_thetas)], dtype=object)  # radius, density, rs, thetas
        else:                                                               # old (probably circle wheel)
            # print("old (probably circle wheel)")
            to_return = np.array([radius, density, None, None], dtype=object)  # radius, density, rs, thetas
    else:                                                                   # old wheel
        # print("old wheel")
        if rs is not None:
            rs_thetas = [(r, theta) for r, theta in zip(rs, thetas)]
            to_return = np.array([radius, density, *rs_thetas_to_string(rs_thetas)], dtype=object)  # radius, density, rs, thetas
        else:
            to_return = np.array([radius, density, None, None], dtype=object)  # radius, density, rs, thetas

    # print(to_return)
    # print("#"*8)
    return to_return


def mutate_vertices(
        chromosome: np.ndarray,
        gaussian_mutation: np.ndarray,
        mask: np.ndarray,
        max_num_wheels_vertices: int,
        min_wheel_vertices_radius: float,
        max_wheel_vertices_radius: float,
        round_length_vertices_coordinates: int
) -> None:

    new_add_vertices = np.where(
        mask[-2, :],
        np.where(
            gaussian_mutation[-2, :] >= 0,
            1,
            -1
        ),
        0
    )

    # print("new_add_vertices:", new_add_vertices, new_add_vertices.shape)
    # print("OLD CHROMOSOME:", chromosome[-4:, :])
    # print("/"*8)

    wheels_rs_thetas = np.array([
        mutate_vertex(
            add_remove,
            radius,
            density,
            string_to_rs_thetas(rs, thetas),
            max_num_wheels_vertices,
            min_wheel_vertices_radius,
            max_wheel_vertices_radius,
            round_length_vertices_coordinates
        )
        for add_remove, radius, density, rs, thetas in zip(new_add_vertices, chromosome[-4, :], chromosome[-3, :], chromosome[-2, :], chromosome[-1, :])
    ])

    # print(wheels_rs_thetas, wheels_rs_thetas.shape)
    # print("OLD CHROMOSOME:", chromosome[-4:, :])
    # print("NEW CHROMOSOME:", wheels_rs_thetas.T, wheels_rs_thetas.T.shape)

    chromosome[-4:, :] = wheels_rs_thetas.T


def decode_encode_mutate_vertices(chromosome_vertices: np.ndarray, prob_mutation: float) -> np.ndarray:
    wheels_rs_gene = chromosome_vertices[0, :]
    wheels_thetas_gene = chromosome_vertices[1, :]

    wheels_rs_thetas = []
    for i, (wheel_rs_str, wheel_thetas_str) in enumerate(zip(wheels_rs_gene, wheels_thetas_gene)):
        if wheel_rs_str and wheel_thetas_str:
            rs, thetas = string_to_rs_thetas(wheel_rs_str, wheel_thetas_str)

            wheels_rs_thetas.append([
                mutate_vertex_old(
                    r,
                    theta,
                    prob_mutation,
                    max_num_wheels_vertices=get_boxcar_constant("max_num_wheels_vertices"),
                    min_wheel_vertices_radius=get_boxcar_constant("min_wheel_vertices_radius"),
                    max_wheel_vertices_radius=get_boxcar_constant("max_wheel_vertices_radius")
                ) for (r, theta) in zip(rs, thetas)
            ])

        else:
            wheels_rs_thetas.append(None)

    wheels_rs, wheels_thetas = rs_thetas_encoding(wheels_rs_thetas)
    chromosome_vertices = np.concatenate((np.array(wheels_rs), np.array(wheels_thetas)), axis=0).reshape((2, 8))

    return chromosome_vertices


def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float,
                      mu: List[float] = None, sigma: List[float] = None,
                      scale: Optional[float] = None) -> None:
    """
    Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.

    If mu and sigma are defined then the gaussian distribution will be drawn from that,
    otherwise it will be drawn from N(0, 1) for the shape of the individual.
    """

    # chromosome_without_vertices_og = chromosome[:-2, :]
    # chromosome_only_vertices_og = chromosome[-2:, :]

    # Determine which genes will be mutated
    # mutation_array = np.random.random(chromosome_without_vertices.shape) < prob_mutation

    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    mutation_array[-1, :] = mutation_array[-2, :]
    # print("mutation_array:", mutation_array, mutation_array.shape)

    # chromosome_to_mutate = chromosome[mutation_array].reshape(7, 8)
    #
    # print(chromosome, chromosome.shape)
    # print(chromosome_to_mutate, chromosome_to_mutate.shape)
    #
    # chromosome_without_vertices = chromosome_to_mutate[:-2, :]
    # chromosome_only_vertices = chromosome_to_mutate[-2:, :]

    # If mu and sigma are defined, create gaussian distribution around each one
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # Otherwise center around N(0,1)
    else:
        # gaussian_mutation = np.random.normal(size=chromosome_without_vertices.shape)
        gaussian_mutation = np.random.normal(size=chromosome.shape)

    if scale:
        gaussian_mutation[mutation_array] *= scale

    gaussian_mutation[-1, :] = gaussian_mutation[-2, :]

    # print("gaussian mutation:", gaussian_mutation, gaussian_mutation.shape)
    # print("chromosome:", chromosome)

    chromosome_mutated_without_vertices = np.where(
        mutation_array[:-2, :],
        (chromosome[:-2, :] + gaussian_mutation[:-2, :]),
        chromosome[:-2, :]
    )

    chromosome[:-2, :] = chromosome_mutated_without_vertices
    # print("Mutated chromosome:", chromosome)

    chromosome[-4:-3, :][
        chromosome[-4:-3, :] < get_boxcar_constant("min_wheel_radius")
    ] = 0.0  # get_boxcar_constant("min_wheel_radius")
    chromosome[-4:-3, :][
        chromosome[-4:-3, :] > get_boxcar_constant("max_wheel_radius")
    ] = get_boxcar_constant("max_wheel_radius")

    chromosome[-3:-2, :][
        chromosome[-3:-2, :] < get_boxcar_constant("min_wheel_density")
    ] = 0.0  # get_boxcar_constant("min_wheel_density")
    chromosome[-3:-2, :][
        chromosome[-3:-2, :] > get_boxcar_constant("max_wheel_density")
    ] = get_boxcar_constant("max_wheel_density")

    # print(chromosome_without_vertices[-2:-1, :])
    # print(chromosome_without_vertices[-1:, :])
    # print("$"*10)
    # print("chromosome_only_vertices_og:", chromosome_only_vertices_og, chromosome_only_vertices_og.shape)

    mutate_vertices(
        chromosome,
        gaussian_mutation,
        mutation_array,
        max_num_wheels_vertices=get_boxcar_constant("max_num_wheels_vertices"),
        min_wheel_vertices_radius=get_boxcar_constant("min_wheel_vertices_radius"),
        max_wheel_vertices_radius=get_boxcar_constant("max_wheel_vertices_radius"),
        round_length_vertices_coordinates=get_boxcar_constant("round_length_vertices_coordinates")
    )

    # print(chromosome)