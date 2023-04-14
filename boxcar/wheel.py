from Box2D import *
from settings import get_boxcar_constant
from typing import List


def calc_polygon_radius_center(vertices: List[b2Vec2]):
    center = sum(vertices, b2Vec2(0, 0)) / len(vertices)

    radiuses = []
    for vertex in vertices:
        radiuses.append((vertex - center).length)

    radius = max(radiuses)
    return radius, center


class Wheel(object):
    def __init__(self, world: b2World, radius: float, density: float, restitution: float = 0.2, vertices: any = None):
        self.radius = radius
        self.density = density
        # self.motor_speed = motor_speed  # Used when it's connected to a chassis
        self.restitution = restitution
        self.vertices = vertices

        # Create body def
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody
        body_def.position = b2Vec2(0, 1)
        self.body = world.CreateBody(body_def)

        if not self.vertices:
            fixture_def = self.circle_wheel()
        else:
            self.radius, self.center = calc_polygon_radius_center(self.vertices)
            fixture_def = self.polygon_wheel(self.vertices)

        # Create fixture on body
        self.body.CreateFixture(fixture_def)

        self._mass = self.body.mass
        self._torque = 0.0

        self.contacts = 0

    def circle_wheel(self):
        # Create fixture def + circle for wheel
        fixture_def = b2FixtureDef()
        circle = b2CircleShape()
        circle.radius = self.radius
        fixture_def.shape = circle
        fixture_def.density = self.density
        fixture_def.friction = 10.0
        fixture_def.restitution = self.restitution
        fixture_def.groupIndex = -1

        return fixture_def

    def polygon_wheel(self, vertices: List[b2Vec2]):
        fixture_def = b2FixtureDef()
        polygon = b2PolygonShape()
        polygon.radius = 0  #self.radius  # MODIFIED added by us TODO check if it works
        fixture_def.shape = polygon
        fixture_def.density = self.density
        fixture_def.friction = 10.0
        fixture_def.restitution = self.restitution
        fixture_def.groupIndex = -1
        fixture_def.shape.vertices = vertices

        return fixture_def

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        raise Exception('Wheel mass is read-only. If you need to change it, do so through Wheel.body.mass or Wheel._mass')

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, value):
        self._torque = value


def clone(self) -> 'Wheel':
    clone = Wheel(self.world, self.radius, self.density, self.restitution, self.vertices)
    return clone