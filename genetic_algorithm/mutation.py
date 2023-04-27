import sys
import numpy as np
from typing import List, Union, Optional
from .individual import Individual
from settings import get_boxcar_constant, get_ga_constant
from encode_decode_chromosome_vertices import pol2b2Vec2, wheels_vertices_pol_to_wheels_vertices, rs_thetas_encoding, rs_thetas_to_string, string_to_rs_thetas

min_wheel_vertices_radius = get_boxcar_constant("min_wheel_vertices_radius")
max_wheel_vertices_radius = get_boxcar_constant("max_wheel_vertices_radius")
max_num_wheels_vertices = get_boxcar_constant("max_num_wheels_vertices")


def mutate_vertex(r: float, theta: int, prob_mutation: float) -> tuple:
    if np.random.random() <= prob_mutation:
        r += (
                (np.random.random() * (max_wheel_vertices_radius - min_wheel_vertices_radius)) + min_wheel_vertices_radius
             ) / (max_wheel_vertices_radius - min_wheel_vertices_radius)

    if np.random.random() <= prob_mutation:
        theta += np.random.randint(low=-(360/max_num_wheels_vertices), high=(360/max_num_wheels_vertices))

    return r, theta


def decode_encode_mutate_vertices(chromosome_vertices: np.ndarray, prob_mutation: float) -> np.ndarray:
    wheels_rs_gene = chromosome_vertices[0, :]
    wheels_thetas_gene = chromosome_vertices[1, :]

    wheels_rs_thetas = []
    for i, (wheel_rs_str, wheel_thetas_str) in enumerate(zip(wheels_rs_gene, wheels_thetas_gene)):
        if wheel_rs_str and wheel_thetas_str:
            rs, thetas = string_to_rs_thetas(wheel_rs_str, wheel_thetas_str)
            wheels_rs_thetas.append([mutate_vertex(r, theta, prob_mutation) for (r, theta) in zip(rs, thetas)])
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

    chromosome_without_vertices = chromosome[:-2, :]
    chromosome_only_vertices = chromosome[-2:, :]

    # Determine which genes will be mutated
    mutation_array = np.random.random(chromosome_without_vertices.shape) < prob_mutation

    # If mu and sigma are defined, create gaussian distribution around each one
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # Otherwise center around N(0,1)
    else:
        gaussian_mutation = np.random.normal(size=chromosome_without_vertices.shape)

    if scale:
        gaussian_mutation[mutation_array] *= scale

    # Update
    chromosome_without_vertices[mutation_array] += gaussian_mutation[mutation_array]

    # TODO check circle wheels and look if radiuses and densities are less than min_wheel_radius and min_wheel_density, else set to 0
    # print("GAUSSIAN MUTATION:", gaussian_mutation)
    # print("GAUSSIAN MUTATION LAST TWO:", gaussian_mutation[-2:, :])
    #
    # print("chromosome_without_vertices:", chromosome_without_vertices)
    # print("chromosome_without_vertices LAST TWO:", chromosome_without_vertices[-2:, :])

    chromosome_without_vertices[-2:-1, :][
        chromosome_without_vertices[-2:-1, :] <= get_boxcar_constant("min_wheel_radius")
    ] = 0.0  # get_boxcar_constant("min_wheel_radius")
    chromosome_without_vertices[-2:-1, :][
        chromosome_without_vertices[-2:-1, :] > get_boxcar_constant("max_wheel_radius")
        ] = get_boxcar_constant("max_wheel_radius")

    chromosome_without_vertices[-1:, :][
        chromosome_without_vertices[-1:, :] <= get_boxcar_constant("min_wheel_density")
        ] = 0.0  # get_boxcar_constant("min_wheel_density")
    chromosome_without_vertices[-1:, :][
        chromosome_without_vertices[-1:, :] > get_boxcar_constant("max_wheel_density")
    ] = get_boxcar_constant("max_wheel_density")

    # print(chromosome_without_vertices[-2:-1, :])
    # print(chromosome_without_vertices[-1:, :])
    # print("$"*10)

    chromosome_only_vertices = decode_encode_mutate_vertices(chromosome_only_vertices, prob_mutation)
    chromosome = np.concatenate((chromosome_without_vertices, chromosome_only_vertices), axis=0)

