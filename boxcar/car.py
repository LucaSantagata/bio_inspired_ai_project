import pandas as pd
from Box2D import *
import numpy as np
from typing import List, Union
from numpy import random
import random as rand
from settings import get_boxcar_constant, get_ga_constant
from .wheel import *
from genetic_algorithm.individual import Individual
from typing import List, Optional, Union, Dict, Any
import math
import dill as pickle
import os
from encode_decode_chromosome_vertices import pol2b2Vec2, wheels_vertices_pol_to_wheels_vertices, rs_thetas_encoding, rs_thetas_to_string, string_to_rs_thetas


genes = {
    # Gene name              row(s)
    'chassis_vertices_x':    0,
    'chassis_vertices_y':    1,
    'chassis_densities':     2,
    'wheel_radii':           3,
    'wheel_densities':       4,
    "wheels_vertices_r":     5,
    "wheels_vertices_theta":     6,
    # 'wheel_motor_speeds':    7,
}


class Car(Individual):
    def __init__(self, world: b2World, 
                 wheel_radii: List[float], wheel_densities: List[float], wheels_vertices_pol: List[tuple],  # wheel_motor_speeds: List[float],
                 chassis_vertices: List[b2Vec2], chassis_densities: List[float],
                 winning_tile: b2Vec2, lowest_y_pos: float, 
                 lifespan: Union[int, float], id: str, from_chromosome: bool = False) -> None:
                 
        self.id = id

        self.world = world
        self.wheel_radii = wheel_radii
        self.wheel_densities = wheel_densities

        self.wheels_masses = []

        self.wheels_vertices_pol = wheels_vertices_pol

        self.wheels_vertices = []

        if wheels_vertices_pol:
            self.wheels_vertices = wheels_vertices_pol_to_wheels_vertices(self.wheels_vertices_pol)

        self.chassis_vertices = chassis_vertices
        self.chassis_densities = chassis_densities

        self.chassis_mass = 0

        self.winning_tile = winning_tile
        self.lowest_y_pos = lowest_y_pos
        self.lifespan = lifespan
        self.is_winner = False

        # These are set in _init_car
        self.chassis = None 

        self.is_alive = True
        self.frames = 0
        self.max_tries = get_boxcar_constant('car_max_tries')
        self.num_failures = 0
        self.cumulative_stall_time = 0
        self.max_position = -100
        self._destroyed = False

        # GA stuff
        self._chromosome = None
        self._fitness = 0.01

        # If the car is being initialized and is NOT from a chromosome, then you need to initialize the GA settins
        # and encode the chromosome. Otherwise it will be taken car of during the deconding of the chromosome
        if not from_chromosome:
            self._init_ga_settings()
            self._init_car()

    def _init_car(self):
        self.chassis = create_chassis(self.world, self.chassis_vertices, self.chassis_densities)
        
        # Calculate chassis volume
        self.chassis_volume = 0.0
        for fixture in self.chassis.fixtures:
            mass = fixture.massData.mass
            density = fixture.density
            self.chassis_volume += mass / density

        # Create wheels from radius/density
        # Since the radius/density arrays are the same length as the chassis vertices, then if there is a positive
        # value, we say the wheel is at the index for the chassis vertex
        self.wheels = []
        self.ordered_wheels = [None for _ in range(8)]
        self._wheel_attachment_vertices = []

        # for i, (wheel_radius, wheel_density, wheel_motor_speed) in enumerate(zip(self.wheel_radii, self.wheel_densities, self.wheel_motor_speeds)):
        for i, (wheel_radius, wheel_density, wheel_vertices) in enumerate(zip(self.wheel_radii, self.wheel_densities, self.wheels_vertices)):
            # Are both above 0?
            if wheel_radius > 0.0 and wheel_density > 0.0:
                self.wheels.append(Wheel(self.world, wheel_radius, wheel_density, vertices=wheel_vertices)) #wheel_motor_speed)) #TODO aggiungi vertici
                self._wheel_attachment_vertices.append(i)  # The chassis vertex this is going to attach to
                self.ordered_wheels[i] = (self.wheels[-1])

        self.num_wheels = len(self.wheels)

        self.chassis_mass = self.chassis.mass

        # Calculate mass of car
        # self.mass = self.chassis_mass
        self.wheels_mass = 0.0
        for wheel in self.ordered_wheels:
            if wheel != None:
                self.wheels_masses.append(wheel.mass)
                self.wheels_mass += wheel.mass
            else:
                self.wheels_masses.append(None)

        self.mass = self.chassis_mass + self.wheels_mass
            
        # Calculate torque of wheel
        for wheel in self.wheels:
            if wheel.radius > 0.0 and wheel.density > 0.0:  # calc torque if b2CircleShape
                torque = self.mass * abs(self.world.gravity.y) * wheel.radius
            # else:  # calc torque if b2PolygonShape
            #     wheel.radius = calc_polygon_radius_center(wheel.vertices)[0]
            #     torque = self.mass * abs(self.world.gravity.y) * wheel.radius
                wheel.torque = torque

        joint_def = b2RevoluteJointDef()
        for i in range(len(self.wheels)):
            # Grab the chassis that the wheel should be on and anchor it
            chassis_vertex = self.chassis_vertices[self._wheel_attachment_vertices[i]]
            joint_def.localAnchorA = chassis_vertex

            if self.wheels[i].vertices:  # b2PolygonShape # TODO check if density works
                # joint_def.localAnchorB = calc_polygon_radius_center(self.wheels[i].vertices)[1]
                joint_def.localAnchorB = self.wheels[i].center
            else:  # b2CircleShape
                joint_def.localAnchorB = self.wheels[i].body.fixtures[0].shape.pos
      
            # Set the motor torque of the wheel - vroom vroom
            joint_def.maxMotorTorque = self.wheels[i].torque
            joint_def.motorSpeed = -15  # self.wheels[i].motor_speed  # @TODO: Make this random
            joint_def.enableMotor = True
            joint_def.bodyA = self.chassis
            joint_def.bodyB = self.wheels[i].body
            self.world.CreateJoint(joint_def)

        # Calculate volume of wheels
        self.wheels_volume = 0.0
        for wheel in self.wheels:
            mass = wheel.body.fixtures[0].massData.mass
            density = wheel.body.fixtures[0].density
            self.wheels_volume += mass / density

    def _init_ga_settings(self) -> None:
        """
        Basic initialization of the chromosome
        """
        # Initialize the chromosome
        self._init_chromosome()

    def _init_chromosome(self) -> None:
        """
        Initializes the chromosome. Only needs to be call
        """
        self._chromosome = np.empty((len(genes.keys()), 8), dtype=object)  # Genes x vertices
        self.encode_chromosome()

    @classmethod
    def create_car_from_chromosome(cls, world: b2World, winning_tile: b2Vec2, lowest_y_pos: float,
                                   lifespan: Union[int, float], chromosome: np.ndarray, id: str) -> 'Car':
        """
        Creates a car from a chromosome. This is helpful in two areas:
        1. You can just keep a bunch of chromosome references and create a car when you need.
        This helps a lot in memory management for Box2D and performance.
        2. You can replay from chromosomes you save.
        """
        car = Car(world, 
                  None, None, None,  # None,  # Wheel stuff set to None
                  None, None,        # Chassis stuff set to None
                  winning_tile, lowest_y_pos, lifespan, id, from_chromosome=True)
        car._chromosome = np.copy(chromosome)
        car.decode_chromosome()
        return car

    def calculate_fitness(self) -> None:
        """
        Calculate the fitness of an individual at the end of a generation.
        """

        # TODO calc contacts of each wheel and add to fitness
        print(f"CAR's WHEELS ({self.num_wheels}, {self._wheel_attachment_vertices}):", self.ordered_wheels)
        #
        # print("Wheel[0] contacts:", self.wheels[0].body.__getattribute__("contacts"))

        wheels_contacts = np.array([wheel.contacts if wheel else np.inf for wheel in self.ordered_wheels])
        # for i, wheel in enumerate(self.ordered_wheels):
        #     if wheel:
        #         print(f"Wheel[{i}] touched {wheel.contacts} times")

        print(wheels_contacts)

        print("///" * 8)
        # func = get_ga_constant('fitness_function')
        # fitness = func(max(self.max_position, 0.0),
        #                self.num_wheels,
        #                self.chassis_volume,
        #                self.wheels_volume,
        #                self.frames)

        func = get_ga_constant('fitness_function2')
        fitness = func(
            max(self.max_position, 0.0),
            int(self.is_winner),
            self.num_wheels,
            get_boxcar_constant("min_num_wheels"),
            get_ga_constant("max_contacts_penalty"),
            get_ga_constant("contacts_threshold"),
            wheels_contacts,
            self.frames,
            self.chassis_volume,
            self.chassis_mass,
            self.wheels_volume,
            self.wheels_mass,
            self.cumulative_stall_time
        )

        print("max_position:", self.max_position)
        print("is_winner:", self.is_winner)
        print("num_wheels:", self.num_wheels)
        print("wheels_contacts:", wheels_contacts)
        print("frames:", self.frames)
        print("chassis_volume:", self.chassis_volume)
        print("chassis_mass:", self.chassis_mass)
        print("wheels_volume:", self.wheels_volume)
        print("wheels_mass:", self.wheels_mass)
        print("cumulative_stall_time:", self.cumulative_stall_time)

        self._fitness = max(fitness, get_ga_constant("min_fitness_value"))

        print("FITNESS FUNCTION:", fitness)
        
    @property
    def fitness(self) -> float:
        return self._fitness
    
    @fitness.setter
    def fitness(self, val):
        self._fitness = val

    def encode_chromosome(self) -> None:
        """
        Encodes (sets the chromosome) from individual values
        """
        #### Chassis stuff
        self._chromosome[genes['chassis_vertices_x'], :] = np.array([vertex.x for vertex in self.chassis_vertices])
        self._chromosome[genes['chassis_vertices_y'], :] = np.array([vertex.y for vertex in self.chassis_vertices])
        self._chromosome[genes['chassis_densities'], :] = np.array([density for density in self.chassis_densities])

        #### Wheel stuff
        self._chromosome[genes['wheel_radii'], :] = np.array([radius for radius in self.wheel_radii])
        self._chromosome[genes['wheel_densities'], :] = np.array([density for density in self.wheel_densities])

        wheels_vertices_r, wheels_vertices_theta = rs_thetas_encoding(self.wheels_vertices_pol)
        self._chromosome[genes["wheels_vertices_r"], :] = np.array(wheels_vertices_r, dtype=object)
        self._chromosome[genes["wheels_vertices_theta"], :] = np.array(wheels_vertices_theta, dtype=object)

        # self._chromosome[genes['wheel_motor_speeds'], :] = np.array([motor_speed for motor_speed in self.wheel_motor_speeds])

    def decode_chromosome(self) -> None:
        """
        Decodes (gets the values) from the chromosome.
        """
        # be a complete polygon if those begin changing drastically
        # If a chassis already exists, then we are going to delete it
        if self.chassis:
            self._destroy()
            # Reset the flags
            self._destroyed = False
            self.is_winner = False
            self.is_alive = True

        #### Decode chassis
        chassis_vertices: b2Vec2 = []
        # Don't forget to.... unzip your genes...
        for xy_vertex in zip(*self._chromosome[(genes['chassis_vertices_x'], genes['chassis_vertices_y']), :]):
            chassis_vertices.append(b2Vec2(xy_vertex))
        self.chassis_vertices = chassis_vertices
        self.chassis_densities = self._chromosome[genes['chassis_densities'], :]
        
        #### Decode wheel
        self.wheel_radii = self._chromosome[genes['wheel_radii'], :]
        self.wheel_densities = self._chromosome[genes['wheel_densities'], :]
        # self.wheel_motor_speeds = self._chromosome[genes['wheel_motor_speeds'], :]

        self.wheels_vertices_pol = [None for _ in range(8)]

        for idx, (wheel_vertices_r, wheel_vertices_theta) in enumerate(zip(*self._chromosome[(genes['wheels_vertices_r'], genes['wheels_vertices_theta']), :])):
            if (wheel_vertices_r and wheel_vertices_theta) and (wheel_vertices_r != "None" and wheel_vertices_theta != "None"):
                wheel_vertices_r, wheel_vertices_theta = string_to_rs_thetas(wheel_vertices_r, wheel_vertices_theta)
                self.wheels_vertices_pol[idx] = [(r, theta) for (r, theta) in zip(wheel_vertices_r, wheel_vertices_theta)]

        self.wheels_vertices = wheels_vertices_pol_to_wheels_vertices(self.wheels_vertices_pol)

        # Re-create the car based off the new chromosome
        self._init_car()

    @property
    def chromosome(self):
        return self._chromosome

    # def clone(self):
    #     world = self.world
    #     wheels = []
    #     for wheel in self.wheels:
    #         radius = wheel.radius
    #         density = wheel.density
    #         restitution = wheel.restitution
    #         vertices = wheel.vertices
    #         wheels.append(Wheel(world, radius, density, restitution, vertices=vertices))
    #
    #     wheel_attachment_vertices = self._wheel_attachment_vertices[:]
    #     wheels_vertices_pol = self.wheels_vertices_pol[:]
    #     chassis_vertices = self.chassis_vertices[:]
    #     chassis_densities = self.chassis_densities[:]
    #     winning_tile = self.winning_tile
    #     lowest_y_pos = self.lowest_y_pos
    #
    #     return Car(world, wheels, wheel_attachment_vertices, wheels_vertices_pol,
    #              chassis_vertices, chassis_densities,
    #              winning_tile, lowest_y_pos, True)

    def update(self) -> bool:
        """
        Determines where the car currently is in comparison to it's goal.
        Has the car died? Did it win? Etc.
        """

        for wheel in self.wheels:
            if wheel.body.__getattribute__("contacts"):
                wheel.contacts += 1

        if not self.is_alive:
            return False

        self.frames += 1
        current_position = self.position
        # Did we win?
        if current_position.x > self.winning_tile.position.x:
            self.is_winner = True
            self.is_alive = False
            self._destroy()
            print('winnnerr')
            return False
        # If we advanced past our max position, reset failures and max position
        if (current_position.x > self.max_position) and (current_position.y > self.lowest_y_pos) and (self.linear_velocity.x >= .4):
            self.num_failures = 0
            self.max_position = current_position.x
            return True

        # If we have not improved or are going very slow, update failures and destroy if needed
        if current_position.x <= self.max_position or self.linear_velocity.x < .4:
            self.num_failures += 1

        if current_position.y < self.lowest_y_pos:
            self.num_failures += 2

        if self.num_failures > self.max_tries:
            self.is_alive = False
        
        if not self.is_alive and not self._destroyed:
            self._destroy()
            return False
        
        self.cumulative_stall_time += self.num_failures

        return True

    def _destroy(self) -> None:
        """
        Cleans up memory from Box2D.
        If you are familiar with C, think of this as "free"
        """
        self.world.DestroyBody(self.chassis)
        for wheel in self.wheels:
            self.world.DestroyBody(wheel.body)
        self._destroyed = True

    @property
    def linear_velocity(self) -> b2Vec2:
        return self.chassis.linearVelocity

    @linear_velocity.setter
    def linear_velocity(self, value):
        # Not actually read only, but don't allow it to be set
        raise Exception('linear velocity is read only!')

    @property
    def position(self) -> b2Vec2:
        return self.chassis.position

    @position.setter
    def position(self, value):
        raise Exception('position is read only!')


def create_random_car(world: b2World, winning_tile: b2Vec2, lowest_y_pos: float, id: str):
    """
    Creates a random car based off the values found in settings.py under the settings dictionary
    """
    # Create a number of random wheels.
    # Each wheel will have a random radius and density
    num_wheels = random.randint(low=get_boxcar_constant('min_num_wheels'), high=get_boxcar_constant('max_num_wheels') + 1)
    wheels_attachment_vertices = list(range(8))   # What vertices should we attach to?
    random.shuffle(wheels_attachment_vertices)
    wheels_attachment_vertices = wheels_attachment_vertices[:num_wheels]
    wheel_radii = [0.0 for _ in range(8)]
    wheel_densities = [0.0 for _ in range(8)]
    wheels_vertices_pol = [None for _ in range(8)]

    # wheel_motor_speeds = [random.uniform(get_boxcar_constant('min_wheel_motor_speed'), get_boxcar_constant('max_wheel_motor_speed'))
    #                       for _ in range(8)]  # Doesn't matter if this is set. There won't be a wheel if the density OR radius is 0

    min_wheel_vertices_radius = get_boxcar_constant('min_wheel_vertices_radius')
    max_wheel_vertices_radius = get_boxcar_constant('max_wheel_vertices_radius')

    # Assign a random radius/density to vertices found in wheel_verts
    for vert_idx in wheels_attachment_vertices:
        radius = random.uniform(get_boxcar_constant('min_wheel_radius'), get_boxcar_constant('max_wheel_radius'))
        density = random.uniform(get_boxcar_constant('min_wheel_density'), get_boxcar_constant('max_wheel_density'))

        if random.random() >= get_boxcar_constant('circle_wheel_probability'):
            num_vertices = random.randint(get_boxcar_constant('min_num_wheels_vertices'), get_boxcar_constant('max_num_wheels_vertices'))

            wheel_vertices_angles = np.arange(0, 360, 360 // num_vertices)

            round_length_vertices_coordinates = get_boxcar_constant("round_length_vertices_coordinates")
            wheel_vertices_radiuses = np.array(
                [
                    round(
                        (
                                (random.random() * (max_wheel_vertices_radius - min_wheel_vertices_radius)) + min_wheel_vertices_radius
                        ),
                        round_length_vertices_coordinates
                    )
                    for _ in range(num_vertices)
                ]
            )

            wheels_vertices_pol[vert_idx] = [(r, theta) for (r, theta) in zip(wheel_vertices_radiuses, wheel_vertices_angles)]

        # Override the intiial 0.0
        wheel_radii[vert_idx] = radius
        wheel_densities[vert_idx] = density
    
    min_chassis_axis = get_boxcar_constant('min_chassis_axis')
    max_chassis_axis = get_boxcar_constant('max_chassis_axis')

    ####
    # The chassis vertices are on a grid and defined by v0-v7 like so:
    # 
    #             v2
    #              |
    #          v3  |  v1
    #     v4 -------------- v0
    #          v5  |  v7
    #              |
    #             v6
    #
    # V0, V2, V4 and V6 are on an axis, while the V1 is defined somewhere between V0 and V2, V3 is defined somewhere between V2 and V4, etc.
    chassis_vertices = []
    chassis_vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), 0))
    chassis_vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), random.uniform(min_chassis_axis, max_chassis_axis)))
    chassis_vertices.append(b2Vec2(0, random.uniform(min_chassis_axis, max_chassis_axis)))
    chassis_vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), random.uniform(min_chassis_axis, max_chassis_axis)))
    chassis_vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), 0))
    chassis_vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), -random.uniform(min_chassis_axis, max_chassis_axis)))
    chassis_vertices.append(b2Vec2(0, -random.uniform(min_chassis_axis, max_chassis_axis)))
    chassis_vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), -random.uniform(min_chassis_axis, max_chassis_axis)))

    # Now t hat we have our chassis vertices, we need to get a random density for them as well
    densities = []
    for i in range(8):
        densities.append(random.uniform(get_boxcar_constant('min_chassis_density'), get_boxcar_constant('max_chassis_density')))

    return Car(world, 
               wheel_radii, wheel_densities, wheels_vertices_pol,  # wheel_motor_speeds,
               chassis_vertices, densities, 
               winning_tile, lowest_y_pos, 
               lifespan=get_ga_constant('lifespan'), id=id)


def create_random_chassis(world: b2World) -> b2Body:
    min_chassis_axis = get_boxcar_constant('min_chassis_axis')
    max_chassis_axis = get_boxcar_constant('max_chassis_axis')

    vertices = []
    vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), 0))
    vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), random.uniform(min_chassis_axis, max_chassis_axis)))
    vertices.append(b2Vec2(0, random.uniform(min_chassis_axis, max_chassis_axis)))
    vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), random.uniform(min_chassis_axis, max_chassis_axis)))
    vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), 0))
    vertices.append(b2Vec2(-random.uniform(min_chassis_axis, max_chassis_axis), -random.uniform(min_chassis_axis, max_chassis_axis)))
    vertices.append(b2Vec2(0, -random.uniform(min_chassis_axis, max_chassis_axis)))
    vertices.append(b2Vec2(random.uniform(min_chassis_axis, max_chassis_axis), -random.uniform(min_chassis_axis, max_chassis_axis)))

    densities = []
    for i in range(8):
        densities.append(random.uniform(get_boxcar_constant('min_chassis_density'), get_boxcar_constant('max_chassis_density')))

    return create_chassis(world, vertices, densities)


def create_chassis(world: b2World, vertices: List[b2Vec2], densities: List[float]) -> b2Body:
    """
    Creates a chassis to be the body of the car.
    """
    if len(vertices) != len(densities):
        raise Exception('vertices and densities must be same length')

    # Create body definition
    body_def = b2BodyDef()
    body_def.type = b2_dynamicBody
    body_def.position = b2Vec2(0, 2)  # Create at (0,1 so it's slightly above the track)

    body = world.CreateBody(body_def)

    # Create chassis parts for the given vertices
    for i in range(len(vertices)):
        # If we are at the end, grab the first index
        if i == len(vertices)-1:
            end_idx = 0
        else:
            end_idx = i+1
        _create_chassis_part(body, vertices[i], vertices[end_idx], densities[i])

    return body


def _create_chassis_part(body: b2Body, point0: b2Vec2, point1: b2Vec2, density: float) -> None:
    """
    Creates a fixture with a polygon shape and adds it to the body.
    The origin point will be (0, 0) and create a polygon with point0 and point1, creating a triangle
    """
    vertices = [point0, point1, b2Vec2(0, 0)]
    
    fixture_def = b2FixtureDef()
    fixture_def.shape = b2PolygonShape()
    fixture_def.density = density
    fixture_def.friction = 10.0
    fixture_def.restitution = 0.2
    fixture_def.groupIndex = -1
    fixture_def.shape.vertices = vertices
    body.CreateFixture(fixture_def)


def smart_clip(chromosome: np.ndarray) -> None:
    """
    Clips the chassis so you can't have 0 density.
    Bad things happen when you give a car 0 density...
    """
    np.clip(chromosome[genes['chassis_densities'], :],
            get_ga_constant("min_fitness_value"),
            np.inf,
            out=chromosome[genes['chassis_densities'], :])


def save_car(population_folder: str, file_name: str, individual_name: str, car: Individual, current_generation: int, datetime: str) -> None:
    """
    Save a car. This saves one and sometimes two things:
    1. Saves the chromosome representation of the individual
    2. Saves the settings. This is only done once.
    """
    # Make the population folder if it doesn't exist
    # if not os.path.exists(population_folder):
    #     os.makedirs(population_folder)

    fname = os.path.join(population_folder, file_name)
    with open(fname, "a") as file:
        file.write(
            str(current_generation) + "," +
            str(car.fitness) + "," +
            str(car.max_position) + "," +
            str(car.chassis_mass) + "," +
            ",".join([str(wheel_mass) for wheel_mass in car.wheels_masses]) + "," +
            str(car.frames) + "," +
            str(car.is_winner) + "," +
            str(car.cumulative_stall_time) + ","
        )
        np.savetxt(file, np.reshape(car.chromosome, (1, 8*7)), delimiter=',', fmt="%s")
    # np.save(fname, car.chromosome)


def load_car(world: b2World, 
             winning_tile: b2Vec2, lowest_y: float,
             lifespan: Union[int, float],
             population_folder: str, individual_name: str) -> Car:
    """
    Loads a car from a folder. This loads the chromosome.
    """
    chromosome = np.load(os.path.join(population_folder, individual_name))
    car = Car.create_car_from_chromosome(world, winning_tile, lowest_y, lifespan, chromosome)
    return car


def load_cars(world: b2World,
             winning_tile: b2Vec2, lowest_y: float,
             lifespan: Union[int, float],
             population_csv: str) -> Car:
    """
    Loads the cars from a csv. This loads the chromosomes.
    """
    df = pd.read_csv(population_csv).groupby("generation").max()
    genes_list = list(genes.keys())

    df_cars_chromosomes = df[[col for col in df.columns for gene in genes_list if gene in col]]

    cars = []
    for i, row in df_cars_chromosomes.iterrows():

        chromosome = np.empty((len(genes), 8), dtype=object)
        for j, gene in enumerate(genes_list):
            chromosome[j] = np.array(
                list(row[[col for col in df.columns if gene in col]])
            )

        cars.append(
            Car.create_car_from_chromosome(
                world,
                winning_tile,
                lowest_y,
                lifespan,
                chromosome
            )
        )

    return cars