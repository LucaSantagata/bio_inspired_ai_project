from Box2D import *
from typing import List
from settings import get_boxcar_constant
import math
import numpy as np


def rotate_floor_tile(coords: List[b2Vec2], center: b2Vec2, angle: float) -> List[b2Vec2]:
    """
    Rotate a given floor tile by some number of degrees.
    """
    rads = angle * math.pi / 180.0  # Degree to radians
    new_coords: List[b2Vec2] = []
    for coord in coords:
        new_coord = b2Vec2()
        new_coord.x = math.cos(rads)*(coord.x - center.x) - math.sin(rads)*(coord.y - center.y) + center.x
        new_coord.y = math.sin(rads)*(coord.x - center.x) + math.cos(rads)*(coord.y - center.y) + center.y
        new_coords.append(new_coord)

    return new_coords

def create_normal_floor_tile(world: b2World, position: b2Vec2, angle: float) -> b2Body:
    """
    Create a floor tile at some angle
    """
    width = get_boxcar_constant('floor_tile_width')
    height = get_boxcar_constant('floor_tile_height')    

    body_def = b2BodyDef()
    body_def.position = position
    body = world.CreateBody(body_def)

    # Create Fixture
    fixture_def = b2FixtureDef()
    fixture_def.shape = b2PolygonShape()
    fixture_def.friction = 0.5

    # Coordinates of tile
    # p1---------p0
    # |          |
    # p2---------p3

    coords: List[b2Vec2] = []
    coords.append(b2Vec2(0, 0))            # p0
    coords.append(b2Vec2(width, 0))        # p1
    coords.append(b2Vec2(width, -height))  # p2
    coords.append(b2Vec2(0, -height))      # p3
    # Rotate @NOTE: This rotates in reference to p0
    coords = rotate_floor_tile(coords, b2Vec2(0, 0), angle)

    # Set vertices of fixture
    fixture_def.shape.vertices = coords

    body.CreateFixture(fixture_def)
    return body

def create_floor_tile(world: b2World, position: b2Vec2, angle: float) -> b2Body:
    """
    Create a floor tile at some angle
    """
    width = get_boxcar_constant('floor_tile_width')
    height = get_boxcar_constant('floor_tile_height')

    # MN MODIFIED for creating strange tiles
    min_section_tile = get_boxcar_constant('min_num_section_per_tile')
    max_section_tile = get_boxcar_constant('max_num_section_per_tile')
    num_section_tile = np.random.randint(min_section_tile, max_section_tile+1)
    
    body_def = b2BodyDef()
    body_def.position = position
    body = world.CreateBody(body_def)

    # Create Fixture
    fixture_def = b2FixtureDef()
    fixture_def.shape = b2PolygonShape()
    fixture_def.friction = 0.5

    # Coordinates of tile
    # p1---------p0
    # |          |
    # p2---------p3

    coords: List[b2Vec2] = []
    coords.append(b2Vec2(0, 0))            # p0

    if (num_section_tile >= 2):
        angle_in_rad = (math.pi/num_section_tile) # calculate the angle of each section in radians
        angle_section = angle_in_rad
        radius = width/2
    
        for i in range(num_section_tile-1):
            if angle_in_rad > (math.pi/2):
                coords.append(b2Vec2(abs(radius+abs(radius * (math.cos(angle_in_rad)))), abs(radius*math.sin(angle_in_rad))))
            else:
                coords.append(b2Vec2(abs(radius-abs(radius * (math.cos(angle_in_rad)))), abs(radius*math.sin(angle_in_rad))))
            angle_in_rad += angle_section
    
    coords.append(b2Vec2(width, 0))        # p1
    coords.append(b2Vec2(width, -height))  # p2
    coords.append(b2Vec2(0, -height))      # p3
    # Rotate @NOTE: This rotates in reference to p0
    coords = rotate_floor_tile(coords, b2Vec2(0, 0), angle)

    # Set vertices of fixture
    fixture_def.shape.vertices = coords

    body.CreateFixture(fixture_def)
    return body


class Floor(object):
    def __init__(self, world: b2World, seed: int = 0, num_tiles: int = 100):
        self.world = world
        self.seed = seed  # @TODO: Add this to the setting
        self.num_tiles = num_tiles
        self.floor_tiles: List[b2Body] = []
        self.rand = np.random.RandomState(self.seed)  # @NOTE: the floor has it's own random that it references.

        self.floor_creation_type = get_boxcar_constant('floor_creation_type').lower()
        if self.floor_creation_type == 'gaussian':
            self._generate_gaussian_random_floor()
        elif self.floor_creation_type == 'ramp':
            self._generate_ramp()
        elif self.floor_creation_type == 'jagged':
            self._create_jagged_floor()
        elif self.floor_creation_type == 'holes':
            self._create_holes_floor()
        elif self.floor_creation_type == 'walls': 
            self._create_wall_floor() 
        elif self.floor_creation_type == 'flat': 
            self._create_flat_floor() 

        self.lowest_y = 10
        for floor_tile in self.floor_tiles:
            world_coords = [floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[i]) for i in range(4)]
            for coord in world_coords:
                if coord.y < self.lowest_y:
                    self.lowest_y = coord.y

    def destroy(self):
        """
        Destroy the floor.
        If you're familiar with C, think of this as "free"
        """
        for tile in self.floor_tiles:
            self.world.DestroyBody(tile)

    def _generate_gaussian_random_floor(self):
        """
        Helper method for generating a gaussian random floor
        """
        threshold = get_boxcar_constant('tile_gaussian_threshold')
        denominator = get_boxcar_constant('tile_gaussian_denominator')
        mu = get_boxcar_constant('tile_angle_mu')
        std = get_boxcar_constant('tile_angle_std')

        tile_position = b2Vec2(-5, 0)
        #@NOTE: Look in README.md for explanation of the below equation
        for i in range(self.num_tiles):
            numerator = min(i, threshold)
            scale = min(float(numerator) / denominator, 1.0)
            angle = self.rand.normal(mu, std) * scale
            floor_tile = create_floor_tile(self.world, tile_position, angle)
            self.floor_tiles.append(floor_tile)

            t = 1
            if angle < 0:
                t = 0

            # @TODO: Fix this. For whatever reason B2D rearranges the vertices. I should track a point during its creation instead
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[t])
            tile_position = world_coord

        self._create_stopping_zone(tile_position)
        
    def _generate_ramp(self):
        """
        Helper method for generating a ramp
        """
        const_angle = get_boxcar_constant('ramp_constant_angle')
        approach_tiles_needed = get_boxcar_constant('ramp_approach_distance') / get_boxcar_constant('floor_tile_width')
        approach_tiles_needed = math.ceil(approach_tiles_needed)

        # Create the approach
        tile_position = b2Vec2(-5, 0)
        for i in range(approach_tiles_needed):
            floor_tile = create_floor_tile(self.world, tile_position, 0)
            self.floor_tiles.append(floor_tile)
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
            tile_position = world_coord

        last_approach_tile = tile_position

        # Are we using a constant angle for the ramp?
        if const_angle:
            num_ramp_tiles = get_boxcar_constant('ramp_constant_distance') / get_boxcar_constant('floor_tile_width')
            num_ramp_tiles = math.ceil(num_ramp_tiles)

            # Create ramp
            for i in range(num_ramp_tiles):
                floor_tile = create_floor_tile(self.world, tile_position, const_angle)
                self.floor_tiles.append(floor_tile)
                world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
                tile_position = world_coord

        # If not, create the increasing ramp
        else:
            start_angle = get_boxcar_constant('ramp_start_angle')
            increasing_angle = get_boxcar_constant('ramp_increasing_angle')
            max_angle = get_boxcar_constant('ramp_max_angle')
            increasing_type = get_boxcar_constant('ramp_increasing_type').lower()
            current_angle = start_angle
            

            # Create ramp
            while True:
                if increasing_type == 'multiply':
                    next_angle = current_angle * increasing_angle
                elif increasing_type == 'add':
                    next_angle = current_angle + increasing_angle
                else:
                    raise Exception("Unknown 'ramp_increasing_type', '{}'".format(increasing_type))

                # If the next requested angle exceeds our maximum, break
                if next_angle > max_angle:
                    break

                floor_tile = create_floor_tile(self.world, tile_position, current_angle)
                self.floor_tiles.append(floor_tile)
                world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
                tile_position = world_coord

                current_angle = next_angle

        # Create the landing zone
        distance_to_fly = get_boxcar_constant('ramp_distance_needed_to_jump')
        tile_position = b2Vec2(tile_position.x + distance_to_fly, last_approach_tile.y)
        for i in range(10):
            floor_tile = create_floor_tile(self.world, tile_position, 0)
            self.floor_tiles.append(floor_tile)
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
            tile_position = world_coord

        self._create_stopping_zone(tile_position)

    def _create_jagged_floor(self):
        """
        Helper method for creating a jagged floor.
        """
        tile_position = b2Vec2(-5, 0)
        increasing_angle = get_boxcar_constant('jagged_increasing_angle')
        decreasing_angle = -get_boxcar_constant('jagged_decreasing_angle')

        for i in range(get_boxcar_constant('max_floor_tiles')):
            angle = increasing_angle if i % 2 == 1 else decreasing_angle
            floor_tile = create_floor_tile(self.world, tile_position, angle)
            self.floor_tiles.append(floor_tile)

            # You can blame this part of B2D. For Python it rearranges the vertices that I reference...
            t = 1
            if angle < 0:
                t =0
            
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[t])
            tile_position = world_coord

        self._create_stopping_zone(tile_position)

    # MN MODIFIED generate holes floors
    def _create_holes_floor(self):
        tile_position = b2Vec2(-5, 0)
        # create first tiles 
        for i in range(10):
            floor_tile = create_floor_tile(self.world, tile_position, 0)
            self.floor_tiles.append(floor_tile)
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
            tile_position = world_coord

        num_holes = get_boxcar_constant('number_of_holes')
        distance_to_fly = get_boxcar_constant('hole_distance_needed_to_jump')
        incremental_distance = get_boxcar_constant('incremental_distance')

        tile_separations = round(self.num_tiles / num_holes)

        for hole in range(num_holes):
            tile_position = b2Vec2(tile_position.x + distance_to_fly, tile_position.y)
            for tile in range(tile_separations):
                floor_tile = create_floor_tile(self.world, tile_position, 0)
                self.floor_tiles.append(floor_tile)
                world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
                tile_position = world_coord
            distance_to_fly = distance_to_fly + incremental_distance

        self._create_stopping_zone(tile_position)

    def _create_wall_floor(self):  
        tile_position = b2Vec2(-5, 0) 
        # create first tiles 
        for i in range(10): 
            floor_tile = create_floor_tile(self.world, tile_position, 0) 
            self.floor_tiles.append(floor_tile) 
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1]) 
            tile_position = world_coord 
         
        num_walls = get_boxcar_constant('number_of_walls') 
        num_wall_tiles = get_boxcar_constant('number_of_wall_tiles')  
        wall_tile_incremental = get_boxcar_constant('wall_tile_incremental')

        tile_separations = round(self.num_tiles / num_walls)

        for wall in range(num_walls):     
            for wall_tile in range(num_wall_tiles):  
                floor_tile = create_floor_tile(self.world, tile_position, 90)  
                self.floor_tiles.append(floor_tile)  
                world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])  
                # Adjust the tile to the left a bit so they overlap and form a wall  
                tile_position = b2Vec2(world_coord.x - get_boxcar_constant('floor_tile_height'), world_coord.y)  
            num_wall_tiles = num_wall_tiles + wall_tile_incremental
            
            for floor_tile in range(tile_separations): 
                floor_tile = create_floor_tile(self.world, tile_position, 0) 
                self.floor_tiles.append(floor_tile) 
                world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1]) 
                tile_position = world_coord 
              
        self._create_stopping_zone(tile_position)  
  
    def _create_flat_floor(self): 
         
        tile_position = b2Vec2(-5, 0) 
        # create first tiles  
        for i in range(10): 
            floor_tile = create_floor_tile(self.world, tile_position, 0) 
            self.floor_tiles.append(floor_tile) 
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1]) 
            tile_position = world_coord 
 
        for i in range(self.num_tiles): 
            floor_tile = create_floor_tile(self.world, tile_position, 0) 
            self.floor_tiles.append(floor_tile) 
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1]) 
            tile_position = world_coord 
         
        self._create_stopping_zone(tile_position) 
 

    def _create_stopping_zone(self, tile_position: b2Vec2) -> None:
        """
        Creates a stopping zone so that the cars have a flat surface at the end of whatever track they were on.
        """
        max_car_size = (get_boxcar_constant('max_chassis_axis') * 2.0) + (2.0 * get_boxcar_constant('max_wheel_radius'))
        tile_width = get_boxcar_constant('floor_tile_width')
        tiles_needed_before_wall = math.ceil(max_car_size / tile_width)
        additional_landing_zone = 0.0
        additional_tiles_needed = additional_landing_zone / tile_width
        total_tiles_needed = math.ceil(tiles_needed_before_wall + additional_tiles_needed + 1)

        # Create a landing zone
        for i in range(total_tiles_needed):
            floor_tile = create_floor_tile(self.world, tile_position, 0)
            self.floor_tiles.append(floor_tile)
            world_coord = floor_tile.GetWorldPoint(floor_tile.fixtures[0].shape.vertices[1])
            tile_position = world_coord

            if i == tiles_needed_before_wall:
                self.winning_tile = self.floor_tiles[-1]
