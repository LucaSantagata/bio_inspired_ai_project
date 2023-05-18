import pandas as pd
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QScrollArea, QStackedWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFormLayout, QComboBox, QGridLayout, QCheckBox, QRadioButton, QMessageBox
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QPixmap, QImage, QFont, QFontMetricsF
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect, QRectF, QLineF
from typing import Optional, Tuple, List, Dict, Any
import argparse
import dill as pickle
from enum import Enum, unique
from Box2D import *
import random
from boxcar.floor import Floor
from boxcar.car import Car, create_random_car, save_car, load_car, load_cars, smart_clip
from genetic_algorithm.population import Population
from genetic_algorithm.individual import Individual
from genetic_algorithm.crossover import single_point_binary_crossover as SPBX
from genetic_algorithm.mutation import gaussian_mutation
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from settings import get_boxcar_constant, get_ga_constant, get_settings, get_window_constant, update_settings_value
import settings
from windows import SettingsWindow, StatsWindow, draw_border
import os
import sys
import time
from datetime import datetime
import numpy as np
import math
import cv2
import atexit


## Constants ##
scale = 70
default_scale = 70
FPS = get_boxcar_constant("fps")


@unique
class States(Enum):
    FIRST_GEN = 0
    FIRST_GEN_IN_PROGRESS = 1
    NEXT_GEN = 2
    NEXT_GEN_COPY_PARENTS_OVER = 4
    NEXT_GEN_CREATE_OFFSPRING = 5
    REPLAY = 6
    STOP = 7


def draw_circle(painter: QPainter, body: b2Body, local: bool = False, adjust_painter: bool = True) -> None:
    """
    Draws a circle with the given painter.
    """

    if adjust_painter:
        _set_painter_clear(painter, Qt.black)

    for fixture in body.fixtures:
        if isinstance(fixture.shape, b2CircleShape):
            # Set the color of the circle to be based off wheel density
            adjust = get_boxcar_constant(
                'max_wheel_density') - get_boxcar_constant('min_wheel_density')
            # If the min/max are the same you will get 0 adjust. This is to prevent divide by zero.
            if adjust == 0.0:
                hue_ratio = 0.0
            else:
                hue_ratio = (fixture.density -
                             get_boxcar_constant('min_wheel_density')) / adjust
            # Just in case you leave the GA unbounded...
            hue_ratio = min(max(hue_ratio, 0.0), 1.0)
            color = QColor.fromHsvF(hue_ratio, 1., .8)
            painter.setBrush(QBrush(color, Qt.SolidPattern))

            radius = fixture.shape.radius
            if local:
                center = fixture.shape.pos
            else:
                center = body.GetWorldPoint(fixture.shape.pos)

            # Fill circle
            painter.drawEllipse(QPointF(center.x, center.y), radius, radius)

            # Draw line (helps for visualization of how fast and direction wheel is moving)
            _set_painter_solid(painter, Qt.black)
            p0 = QPointF(center.x, center.y)
            p1 = QPointF(center.x + radius*math.cos(body.angle),
                         center.y + radius*math.sin(body.angle))
            painter.drawLine(p0, p1)


def draw_polygon(painter: QPainter, body: b2Body, poly_type: str = '', adjust_painter: bool = True, local=False) -> None:
    """
    Draws a polygon with the given painter. Uses poly_type for determining the fill of the polygon.
    """
    if adjust_painter:
        _set_painter_clear(painter, Qt.black)

    for fixture in body.fixtures:
        if isinstance(fixture.shape, b2PolygonShape):
            poly = []
            # If we are drawing a chassis, determine fill color
            if poly_type == 'chassis':
                adjust = get_boxcar_constant(
                    'max_chassis_density') - get_boxcar_constant('min_chassis_density')
                # If the min/max are the same you will get 0 adjust. This is to prevent divide by zero.
                if adjust == 0.0:
                    hue_ratio = 0.0
                else:
                    hue_ratio = (
                        fixture.density - get_boxcar_constant('min_chassis_density')) / adjust
                # Just in case you leave the GA unbounded...
                hue_ratio = min(max(hue_ratio, 0.0), 1.0)
                color = QColor.fromHsvF(hue_ratio, 1., .8)
                painter.setBrush(QBrush(color, Qt.SolidPattern))
            elif poly_type == 'wheel':
                adjust = get_boxcar_constant(
                    'max_wheel_density') - get_boxcar_constant('min_wheel_density')
                # If the min/max are the same you will get 0 adjust. This is to prevent divide by zero.
                if adjust == 0.0:
                    hue_ratio = 0.0
                else:
                    hue_ratio = (fixture.density -
                                 get_boxcar_constant('min_wheel_density')) / adjust
                # Just in case you leave the GA unbounded...
                hue_ratio = min(max(hue_ratio, 0.0), 1.0)
                color = QColor.fromHsvF(hue_ratio, 1., .8)
                painter.setBrush(QBrush(color, Qt.SolidPattern))

            polygon: b2PolygonShape = fixture.shape
            local_points: List[b2Vec2] = polygon.vertices

            if local:
                world_coords = local_points
            else:
                world_coords = [body.GetWorldPoint(
                    point) for point in local_points]
            for i in range(len(world_coords)):
                p0 = world_coords[i]
                if i == len(world_coords)-1:
                    p1 = world_coords[0]
                else:
                    p1 = world_coords[i+1]

                qp0 = QPointF(*p0)
                qp1 = QPointF(*p1)

                poly.append(qp0)
                poly.append(qp1)
            if poly:
                painter.drawPolygon(QPolygonF(poly))


def draw_label(painter: QPainter, car: Car, adjust_painter: bool = True) -> None:
    """
    Draws a label on top of the car
    """
    if car.is_alive:
        if adjust_painter:
            _set_painter_clear(painter, Qt.black, scale=7)

        font = QFont('Arial', 1, QFont.Light)
        text_scale = 10
        painter.scale(-1/text_scale, 1/text_scale)
        painter.setFont(font)
        label_text = str(car.id)
        # get the bounding rectangle of the text
        fm = QFontMetricsF(font)
        text_width = fm.width(label_text)
        text_height = fm.height()
        # adjust the rectangle size to fit the text
        offset = 1
        point = car.chassis.GetWorldPoint((0,0)) # get the center of the car
        label_rect = QRectF(point[0] * text_scale - (text_width + offset)/2 , point[1] * (-text_scale) - 15,  text_width + offset, text_height + 2*offset) # create a rectangle around the text
        painter.rotate(180) # rotate the painter so the text is not upside down
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern)) # set the brush to white for filling the rectangle
        painter.drawRect(label_rect) # draw the rectangle
        painter.drawText(label_rect, Qt.AlignCenter, label_text) # draw the text
        line = QLineF(point[0]* text_scale, point[1]* (-text_scale), label_rect.center().x(), label_rect.center().y()+ 2*offset)  # create the line here so it can be dynamic 
        painter.drawLine(line) # draw the line
        painter.rotate(-180) # rotate the painter back to normal
        painter.scale(-1*text_scale, 1*text_scale) # scale the painter back to normal


def _set_painter_solid(painter: QPainter, color: Qt.GlobalColor, with_antialiasing: bool = True, scale: int = scale):
    _set_painter(painter, color, True, with_antialiasing, scale)


def _set_painter_clear(painter: QPainter, color: Qt.GlobalColor, with_antialiasing: bool = True, scale: int = scale):
    _set_painter(painter, color, False, with_antialiasing, scale)


def _set_painter(painter: QPainter, color: Qt.GlobalColor, fill: bool, with_antialiasing: bool = True, scale: int = scale):
    painter.setPen(QPen(color, 1./scale, Qt.SolidLine))
    pattern = Qt.SolidPattern if fill else Qt.NoBrush
    painter.setBrush(QBrush(color, pattern))
    if with_antialiasing:
        painter.setRenderHint(QPainter.Antialiasing)


class InitWindow(QWidget):
    def __init__(self, stacked_window, output_path, _datetime, replay):
        super().__init__()

        self.initUI()

        self.floor_type = None
        self.gravity = None
        self.tiles_type = None

        self.stacked_window = stacked_window

        self.output_path = output_path
        self._datetime = _datetime
        self.replay = replay

    def initUI(self):
        # Create combo box for floors
        floor_label = QLabel('Floor type')
        self.floor_combo = QComboBox(self)
        self.floor_combo.addItem("Gaussian")
        self.floor_combo.addItem("Ramp")
        self.floor_combo.addItem("Jagger")
        self.floor_combo.addItem("Holes")
        self.floor_combo.move(50, 50)

        # Create combo box for gravities
        gravity_label = QLabel('Available gravities')
        self.gravity_combo = QComboBox(self)
        self.gravity_combo.addItem("Earth") # -9.81
        self.gravity_combo.addItem("Mars") # -3.711
        self.gravity_combo.addItem("Moon") # -1.622
        self.gravity_combo.move(50, 100)

        # Create combo box for tiles
        tiles_label = QLabel('Tiles type')
        self.tiles_combo = QComboBox(self)
        self.tiles_combo.addItem("Straight Line")
        self.tiles_combo.addItem("Circle")
        self.tiles_combo.addItem("Triangle")
        self.tiles_combo.addItem("Random Polygon")
        self.tiles_combo.move(50, 150)

        # Create check boxes
        self.save_pop_check = QCheckBox('Save population')
        self.save_pop_line = QLineEdit()
        self.save_pop_line.setPlaceholderText('Enter your dir path here')
        self.save_pop_line.setText("./savepop/")

        self.save_video_check = QCheckBox('Save video')
        self.save_video_line = QLineEdit()
        self.save_video_line.setPlaceholderText('Enter your dir path here')
        self.save_video_line.setText("./video/")

        self.none_check = QRadioButton('None')

        self.replay_check = QRadioButton('Replay')
        self.replay_line = QLineEdit()
        self.replay_line.setPlaceholderText('Enter your file path here')

        self.test_check = QRadioButton('Test')
        self.test_line = QLineEdit()
        self.test_line.setPlaceholderText('Enter your file path here')

        self.replay_check.toggled.connect(self.check_toggle)
        self.test_check.toggled.connect(self.check_toggle)

        # Create send update button
        set_button = QPushButton('Set parameters', self)
        set_button.clicked.connect(self.set_parameters)

        idx = 0
        gridLayout = QGridLayout()
        gridLayout.addWidget(floor_label, idx, 0)
        gridLayout.addWidget(self.floor_combo, idx, 1)

        idx += 1
        gridLayout.addWidget(gravity_label, idx, 0)
        gridLayout.addWidget(self.gravity_combo, idx, 1)

        idx += 1
        gridLayout.addWidget(tiles_label, idx, 0)
        gridLayout.addWidget(self.tiles_combo, idx, 1)

        idx += 1
        gridLayout.addWidget(self.save_pop_check, idx, 0)
        gridLayout.addWidget(self.save_pop_line, idx, 1)

        idx += 1
        gridLayout.addWidget(self.save_video_check, idx, 0)
        gridLayout.addWidget(self.save_video_line, idx, 1)

        idx += 1
        gridLayout.addWidget(self.none_check, idx, 0)

        idx += 1
        gridLayout.addWidget(self.replay_check, idx, 0)
        gridLayout.addWidget(self.replay_line, idx, 1)

        idx += 1
        gridLayout.addWidget(self.test_check, idx, 0)
        gridLayout.addWidget(self.test_line, idx, 1)

        idx += 1
        gridLayout.addWidget(set_button, idx, 0)
        self.setLayout(gridLayout)

        # Imposta le dimensioni della finestra
        self.setGeometry(0, 0, get_window_constant('width'), get_window_constant('height'))
        self.setWindowTitle('Setting window')
        self.show()

    def check_toggle(self):
        if self.replay_check.isChecked() or self.test_check.isChecked():
            self.save_pop_check.setEnabled(False)
            self.save_pop_line.setEnabled(False)
        else:
            self.save_pop_check.setEnabled(True)
            self.save_pop_line.setEnabled(True)

    def set_parameters(self):
        print("Set parameters")

        self.floor_type = self.floor_combo.currentText()
        self.gravity = self.gravity_combo.currentText()
        self.tiles_type = self.tiles_combo.currentText()

        if self.floor_type == 'Gaussian':
            update_settings_value("boxcar", "floor_creation_type", ("gaussian", str))
        elif self.floor_type == 'Ramp':
            update_settings_value("boxcar", "floor_creation_type", ("ramp", str))
        elif self.floor_type == 'Jagger':
            update_settings_value("boxcar", "floor_creation_type", ("jagger", str))
        elif self.floor_type == 'Holes':
            update_settings_value("boxcar", "floor_creation_type", ("holes", str))

        if self.gravity == 'Earth':
            update_settings_value("boxcar", "gravity", ((0, -9.81), tuple))
        elif self.gravity == 'Mars':
            update_settings_value("boxcar", "gravity", ((0, -3.711), tuple))
        elif self.gravity == 'Moon':
            update_settings_value("boxcar", "gravity", ((0, -1.622), tuple))

        if self.tiles_type == 'Straight Line':
            update_settings_value("boxcar", "min_num_section_per_tile", (1, int))
            update_settings_value("boxcar", "max_num_section_per_tile", (1, int))
        elif self.tiles_type == 'Circle':
            update_settings_value("boxcar", "min_num_section_per_tile", (10, int))
            update_settings_value("boxcar", "max_num_section_per_tile", (10, int))
        elif self.tiles_type == 'Triangle':
            update_settings_value("boxcar", "min_num_section_per_tile", (2, int))
            update_settings_value("boxcar", "max_num_section_per_tile", (2, int))
        elif self.tiles_type == 'Random Polygon':
            update_settings_value("boxcar", "min_num_section_per_tile", (3, int))
            update_settings_value("boxcar", "max_num_section_per_tile", (8, int))

        if (
            (
                self.save_pop_line.text() == '' and
                self.save_pop_check.isChecked() and
                self.save_pop_check.isEnabled()
            ) or (
                    self.save_video_line.text() == '' and
                    self.save_video_check.isChecked()
            ) or (
                    self.replay_line.text() == '' and
                    self.replay_check.isChecked()
            ) or (
                    self.test_line.text() == '' and
                    self.test_check.isChecked()
            )
        ):
            message = QMessageBox()
            message.setWindowTitle('Error')
            message.setText('Please fill in all fields.')
            message.setIcon(QMessageBox.Critical)
            message.exec_()
        else:
            if self.replay_check.isChecked():
                args.replay_from_filename = self.replay_line.text()
                self.replay = True
            elif self.test_check.isChecked():
                args.test_from_filename = self.test_line.text()
                self.replay = True

            if self.save_pop_check.isChecked() and self.save_pop_check.isEnabled():
                args.save_pop = self.save_pop_line.text()
            if self.save_video_check.isChecked():
                args.save_video = self.save_video_line.text()

                if not os.path.exists(args.save_video):
                    # raise Exception('{} already exists. This would overwrite everything, choose a different folder or delete it and try again'.format(path))
                    os.makedirs(args.save_video)

            world = b2World(get_boxcar_constant('gravity'))

            if args.save_video:
                output_file = "video_" + _datetime + ".mp4"
                self.output_path = os.path.join(args.save_video, output_file)

            self.window = MainWindow(world, self.output_path, self._datetime, self.replay)
            self.stacked_window.addWidget(self.window)
            self.stacked_window.move(0, 0)
            self.stacked_window.setFixedWidth(get_window_constant('width'))
            self.stacked_window.setFixedHeight(get_window_constant('height'))
            self.stacked_window.setCurrentWidget(self.window)


class GameWindow(QWidget):
    def __init__(self, parent, size, world, floor, cars, leader):
        super().__init__(parent)
        self.size = size
        self.world = world
        self.title = 'Test'
        self.top = 0
        self.left = 0

        self.floor = floor
        self.leader: Car = leader  # Track the leader
        self.best_car_ever = None
        self.cars = cars
        self.manual_control = False  # W,A,S,D, Z,C, E,R

        # Camera stuff
        self._camera = b2Vec2()
        self._camera_speed = 0.05
        self._camera.x

    def pan_camera_to_leader(self, should_smooth: bool = False) -> None:
        if should_smooth:
            diff_x = self._camera.x - self.leader.chassis.position.x
            diff_y = self._camera.y - self.leader.chassis.position.y
            self._camera.x -= self._camera_speed * diff_x
            self._camera.y -= self._camera_speed * diff_y
        else:
            self._camera.x = self.leader.chassis.position.x
            self._camera.y = self.leader.chassis.position.y

    def pan_camera_in_direction(self, direction: str, amount: int) -> None:
        diff_x, diff_y = 0, 0
        if direction.lower()[0] == 'u':
            diff_y = -amount
        elif direction.lower()[0] == 'd':
            diff_y = amount
        elif direction.lower()[0] == 'l':
            diff_x = amount
        elif direction.lower()[0] == 'r':
            diff_x = -amount

        self._camera.x -= self._camera_speed * diff_x
        self._camera.y -= self._camera_speed * diff_y

    def _update(self):
        """
        Main update method used. Called once every (1/FPS) second.
        """
        self.update()

    def _draw_car(self, painter: QPainter, car: Car):
        """
        Draws a car to the window
        """
        for wheel in car.wheels:
            if wheel.vertices:
                draw_polygon(painter, wheel.body, poly_type='wheel')
            else:
                draw_circle(painter, wheel.body)

        draw_polygon(painter, car.chassis, poly_type='chassis')

        if get_boxcar_constant('show_label'):
            # draw the label of the car
            draw_label(painter, car)

    def _draw_floor(self, painter: QPainter):
        # @TODO: Make this more efficient. Only need to draw things that are currently on the screen or about to be on screen
        for tile in self.floor.floor_tiles:
            if tile is self.floor.winning_tile:
                painter.setPen(QPen(Qt.black, 1./scale, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
                painter.setRenderHint(QPainter.Antialiasing)
                local_points: List[b2Vec2] = tile.fixtures[0].shape.vertices
                world_coords = [tile.GetWorldPoint(
                    point) for point in local_points]
                qpoints = [QPointF(coord[0], coord[1])
                           for coord in world_coords]
                polyf = QPolygonF(qpoints)
                painter.drawPolygon(polyf)
            else:
                draw_polygon(painter, tile)

    def paintEvent(self, event):
        painter = QPainter(self)
        draw_border(painter, self.size)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.translate(200 - (self._camera.x * scale),
                          250 + (self._camera.y * scale))
        # painter.translate(200,300)
        painter.scale(scale, -scale)
        arr = [Qt.black, Qt.green, Qt.blue]
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))

        self._draw_floor(painter)

        # self.draw_polygon(painter, self.chassis)
        for car in self.cars:
            self._draw_car(painter, car)
        # for fixture in self.chassis.fixtures:
        #     print([self.chassis.GetWorldPoint(vert) for vert in fixture.shape.vertices])


class MainWindow(QMainWindow):
    def __init__(self, world, video_path, _datetime, replay: bool = False, run: str = ""):
        super().__init__()

        self.datetime = _datetime
        self.run = run
        self.file_name = self.datetime + "_" + self.run + ".csv"
        if video_path is not None:
            self.video_path = video_path[:video_path.index(".m")] + "_run" + self.run + video_path[video_path.index(".m"):]
        else:
            self.video_path = None

        self.world = world
        self.title = 'Genetic Algorithm - Cars'
        self.top = 0
        self.left = 0

        self.width = get_window_constant('width')
        self.height = get_window_constant('height')

        self.max_fitness = 0.0
        self.cars = []
        self.population = Population([])
        self.state = States.FIRST_GEN
        # Used when you are in state 1, i.e. creating new cars from the old population
        self._next_pop = []
        self.current_batch = 1
        self.batch_size = get_boxcar_constant('run_at_a_time')
        self.gen_without_improvement = 0
        self.replay = replay

        self.out = None
        if self.video_path is not None:
            self.out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (self.width, self.height))
            self.out.set(cv2.CAP_PROP_BITRATE, 1000)  # reduce bitrate to reduce file size # 500 maybe

        self.manual_control = False

        self.current_generation = 0
        self.leader = None  # What car is leading
        self.num_cars_alive = get_boxcar_constant('run_at_a_time')
        self.batch_size = self.num_cars_alive
        self._total_individuals_ran = 0
        self._offset_into_population = 0  # Used if we display only a certain number at a
        # Determine whether or not we are in the process of creating random cars.
        # This is used for when we only run so many at a time. For instance if `run_at_a_time` is 20 and
        # `num_parents` is 1500, then we can't just create 1500 cars. Instead we create batches of 20 to
        # run at a time. This flag is for deciding when we are done with that so we can move on to crossover
        # and mutation.
        self._creating_random_cars = True
        self._all_gen_winners = 0

        self.num_car_generated = 0

        # Determine how large the next generation is
        if get_ga_constant('selection_type').lower() == 'plus':
            self._next_gen_size = get_ga_constant(
                'num_parents') + get_ga_constant('num_offspring')
        elif get_ga_constant('selection_type').lower() == 'comma':
            self._next_gen_size = get_ga_constant('num_parents')
        else:
            raise Exception('Selection type "{}" is invalid'.format(
                get_ga_constant('selection_type')))

        if self.replay:
            global args
            self.floor = Floor(self.world, seed=get_boxcar_constant('gaussian_floor_seed'), num_tiles=get_boxcar_constant('max_floor_tiles'))
            self.state = States.REPLAY

            # self.num_replay_inds = len([x for x in os.listdir(args.replay_from_folder) if x.startswith('car_')])
            self.num_replay_inds = 1
        else:
            self._set_first_gen()
        # self.population = Population(self.cars)
        # For now this is all I'm supporting, may change in the future. There really isn't a reason to use
        # uniform or single point here because all the values have different ranges, and if you clip them, it
        # can make those crossovers useless. Instead just use simulated binary crossover to ensure better crossover.

        self.init_window()
        self.stats_window.pop_size.setText(str(get_ga_constant('num_parents')))
        self._set_number_of_cars_alive()
        self.game_window.cars = self.cars
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(1000//get_boxcar_constant('fps'))

    def next_generation(self) -> None:
        if self.state == States.NEXT_GEN:
            self.stats_window.pop_size.setText(str(self._next_gen_size))
            self.current_batch = 0
            # Set next state to copy parents if its plus, otherwise comma is just going to create offspring
            if get_ga_constant('selection_type').lower() == 'plus':
                self.state = States.NEXT_GEN_COPY_PARENTS_OVER
            elif get_ga_constant('selection_type').lower() == 'comma':
                self.state = States.NEXT_GEN_CREATE_OFFSPRING
            else:
                raise Exception('Invalid selection_type: "{}"'.format(
                    get_ga_constant('selection_type')))

            self._offset_into_population = 0
            self._total_individuals_ran = 0  # Reset back to the first individual

            self.population.individuals = self._next_pop
            self._next_pop = []  # Reset the next pop

            # Calculate fit
            for individual in self.population.individuals:
                individual.calculate_fitness()

            # Should we save the pop
            if args.save_pop:
                path = args.save_pop

                if not os.path.exists(path):
                    # raise Exception('{} already exists. This would overwrite everything, choose a different folder or delete it and try again'.format(path))
                    os.makedirs(path)
                save_population(path, self.file_name, self.population, get_settings(), self.current_generation, self.datetime, self.run)
            # Save best? 
            if args.save_best:
                save_car(args.save_best, 'car_{}'.format(self.current_generation), self.population.fittest_individual, get_settings(), self.current_generation)

            self._set_previous_gen_avg_fitness()
            self._set_previous_gen_num_winners()
            self._increment_generation()

            # Grab the best individual and compare to best fitness
            best_ind = self.population.fittest_individual
            if best_ind.fitness > self.max_fitness:
                self.max_fitness = best_ind.fitness
                self._set_max_fitness()
                self.gen_without_improvement = 0
            else:
                self.gen_without_improvement += 1
            # Set text for gen improvement
            self.stats_window.gens_without_improvement.setText(
                str(self.gen_without_improvement))

            # Set the population to be just the parents allowed for reproduction. Only really matters if `plus` method is used.
            # If `plus` method is used, there can be more individuals in the next generation, so this limits the number of parents.
            self.population.individuals = elitism_selection(
                self.population, get_ga_constant('elitism'))

            random.shuffle(self.population.individuals)

            # Parents + offspring selection type ('plus')
            if get_ga_constant('selection_type').lower() == 'plus':
                # Decrement lifespan
                for individual in self.population.individuals:
                    individual.lifespan -= 1

        num_offspring = min(self._next_gen_size - len(self._next_pop), get_boxcar_constant('run_at_a_time'))
        self.cars = self._create_num_offspring(num_offspring)
        # Set number of cars alive
        self.num_cars_alive = len(self.cars)
        self.batch_size = self.num_cars_alive
        self.current_batch += 1
        self._set_number_of_cars_alive()
        self._next_pop.extend(self.cars)  # Add to next_pop
        self.game_window.cars = self.cars
        leader = self.find_new_leader()
        self.leader = leader
        self.game_window.leader = leader
        if get_ga_constant('selection_type').lower() == 'comma':
            self.state = States.NEXT_GEN_CREATE_OFFSPRING
        elif get_ga_constant('selection_type').lower() == 'plus' and self._offset_into_population >= len(self.population.individuals):
            self.state = States.NEXT_GEN_CREATE_OFFSPRING

        # Set the next pop
        # random.shuffle(next_pop)
        # self.population.individuals = next_pop

    def get_id(self):
        """
        Sets the id of the car based on the current generation and the number of cars generated
        """
        # id_text = f'Gen_{self.current_generation+1}_id_{self.num_car_generated}'
        id_text = f'{self.current_generation+1}_{self.num_car_generated}' # second way to set id, which is maybe cleaner
        self.num_car_generated += 1
        return id_text

    def init_window(self):
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self._game_window_sizes = {  # up left
            "x": 0,
            "y": 0,
            "w": 1500,
            "h": 700
        }

        self._stats_window_sizes = {  # bottom left
            "x": 0,
            "y": self._game_window_sizes["y"] + self._game_window_sizes["h"],
            "w": self._game_window_sizes["w"],
            "h": self.height - self._game_window_sizes["h"]
        }

        self._settings_window_sizes = {  # top right
            "x": self._game_window_sizes["x"] + self._game_window_sizes["w"],
            "y": 0,
            "w": self.width - self._game_window_sizes["w"],
            "h": self.height
        }

        # Create game_window - where the game is played
        self.game_window = GameWindow(
            self.centralWidget,
            (self._game_window_sizes["w"], self._game_window_sizes["h"]),
            self.world, self.floor, self.cars, self.leader
        )
        self.game_window.setGeometry(QRect(
            self._game_window_sizes["x"], self._game_window_sizes["y"], self._game_window_sizes["w"], self._game_window_sizes["h"]
        ))
        self.game_window.setObjectName("game_window")

        # Create stats_window
        self.stats_window = StatsWindow(
            self.centralWidget,
            (self._stats_window_sizes["w"], self._stats_window_sizes["h"])
        )
        self.stats_window.setGeometry(QRect(
            self._stats_window_sizes["x"], self._stats_window_sizes["y"],
            self._stats_window_sizes["w"], self._stats_window_sizes["h"]
        ))
        self.stats_window.setObjectName('stats_window')

        # Create settings_window - just a bunch of settings of the game and how they were defined, etc.
        self.settings_window = SettingsWindow(
            self.centralWidget,
            (self._settings_window_sizes["x"],
             self._settings_window_sizes["y"])
        )
        self.settings_window.setGeometry(QRect(
            self._settings_window_sizes["x"], self._settings_window_sizes["y"],
            self._settings_window_sizes["w"], self._settings_window_sizes["h"]
        ))
        self.settings_window.setObjectName('settings_window')

        # Add main window
        self.main_window = QWidget(self)
        self.main_window.setGeometry(QRect(0, 0, self.width, self.height))
        self.main_window.setObjectName('main_window')

        if get_boxcar_constant('show'):
            self.show()

    def find_new_leader(self) -> Optional[Car]:
        max_x = -1
        leader = None
        for car in self.cars:
            # Can't be a leader if you're dead
            if not car.is_alive:
                continue

            car_pos = car.position.x
            if car_pos > max_x:
                leader = car
                max_x = car_pos

        return leader

    def _set_previous_gen_avg_fitness(self) -> None:
        avg_fitness = sum(
            ind.fitness for ind in self.population.individuals) / len(self.population.individuals)
        # self.stats_window.average_fitness_last_gen.setText('{:.5f}'.format(avg_fitness))
        self.stats_window.average_fitness_last_gen.setText(str(avg_fitness))

    def _set_previous_gen_num_winners(self) -> None:
        winners = sum(ind.is_winner for ind in self.population.individuals)
        self._all_gen_winners += winners
        self.stats_window.num_solved_last_gen.setText(str(winners))
        self.stats_window.num_cum_solved_gen.setText(str(self._all_gen_winners))

    def _create_num_offspring(self, number_of_offspring) -> List[Individual]:
        """
        This is a helper function to decide whether to grab from current pop or create new offspring.

        Creates a number of offspring from the current population. This assumes that the current population are all able to reproduce.
        This is broken up from the main next_generation function so that we can create N individuals at a time if needed without going
        to the next generation. Mainly used if `run_at_a_time` is < the number of individuals that are in the next generation.
        """
        next_pop: List[Individual] = []
        # @TODO: comment this to new state
        # If the selection type is plus, then it means certain individuals survive to the next generation, so we need
        # to grab those first before we create new ones
        # if get_ga_constant('selection_type').lower() == 'plus' and len(self._next_pop) < get_ga_constant('num_parents'):
        if self.state == States.NEXT_GEN_COPY_PARENTS_OVER:
            # Select the subset of the individuals to bring to the next gen
            increment = 0  # How much did the offset increment by
            for idx in range(self._offset_into_population, len(self.population.individuals)):
                # for individual in self.population.individuals[self._offset_into_population: self._offset_into_population + number_of_offspring]:
                individual = self.population.individuals[idx]
                increment += 1  # For offset
                world = self.world
                wheel_radii = individual.wheel_radii
                wheel_densities = individual.wheel_densities
                wheels_vertices_pol = individual.wheels_vertices_pol
                # wheel_motor_speeds = individual.wheel_motor_speeds
                chassis_vertices = individual.chassis_vertices
                chassis_densities = individual.chassis_densities
                winning_tile = individual.winning_tile
                lowest_y_pos = individual.lowest_y_pos
                lifespan = individual.lifespan

                # If the individual is still alive, they survive
                if lifespan > 0:
                    car = Car(world,
                              # wheel_motor_speeds,       # Wheel #TODO add vertices
                              wheel_radii, wheel_densities, wheels_vertices_pol,
                              chassis_vertices, chassis_densities,                    # Chassis
                              winning_tile, lowest_y_pos,
                              lifespan, self.get_id())
                    next_pop.append(car)
                    # Check to see if we've added enough parents. The reason we check here is if you requet 5 parents but
                    # 2/5 are dead, then you need to keep going until you get 3 good ones.
                    if len(next_pop) == number_of_offspring:
                        break
                else:
                    print("Oh dear, you're dead")
            # Increment offset for the next time
            self._offset_into_population += increment
            # If there weren't enough parents that made it to the new generation, we just accept it and move on.
            # Since the lifespan could have reached 0, you are not guaranteed to always have the same number of parents copied over.
            if self._offset_into_population >= len(self.population.individuals):
                self.state = States.NEXT_GEN_CREATE_OFFSPRING
        # Otherwise just perform crossover with the current population and produce num_of_offspring
        # @NOTE: The state, even if we got here through State.NEXT_GEN or State.NEXT_GEN_COPY_PARENTS_OVER is now
        # going to switch to State.NEXT_GEN_CREATE_OFFSPRING based off this else condition. It's not set here, but
        # rather at the end of new_generation
        else:
            # Keep adding children until we reach the size we need
            while len(next_pop) < number_of_offspring:
                # Tournament crossover
                if get_ga_constant('crossover_selection').lower() == 'tournament':
                    p1, p2 = tournament_selection(
                        self.population, 2, get_ga_constant('tournament_size'))
                # Roulette
                elif get_ga_constant('crossover_selection').lower() == 'roulette':
                    p1, p2 = roulette_wheel_selection(self.population, 2)
                else:
                    raise Exception('crossover_selection "{}" is not supported'.format(
                        get_ga_constant('crossover_selection').lower()))

                # Crossover
                c1_chromosome, c2_chromosome = self._crossover(
                    p1.chromosome, p2.chromosome)

                # Mutation
                self._mutation(c1_chromosome)
                self._mutation(c2_chromosome)

                # Don't let the chassis density become <=0. It is bad
                smart_clip(c1_chromosome)
                smart_clip(c2_chromosome)

                # Create children from the new chromosomes
                c1 = Car.create_car_from_chromosome(
                    p1.world, p1.winning_tile, p1.lowest_y_pos, get_ga_constant('lifespan'), c1_chromosome, self.get_id())
                c2 = Car.create_car_from_chromosome(
                    p2.world, p2.winning_tile, p2.lowest_y_pos, get_ga_constant('lifespan'), c2_chromosome, self.get_id())

                # Add children to the next generation
                next_pop.extend([c1, c2])

        # Return the next population that will play. Remember, this can be a subset of the overall population since
        # those parents still exist.
        return next_pop

    def _increment_generation(self) -> None:
        """
        Increments the generation and sets the label
        """
        self.current_generation += 1
        self.stats_window.generation.setText(
            "<font color='red'>" + str(self.current_generation + 1) + '</font>')

    def _set_first_gen(self) -> None:
        """
        Sets the first generation, i.e. random cars
        """
        # Create the floor if FIRST_GEN, but not if it's in progress
        if self.state == States.FIRST_GEN:
            self.floor = Floor(self.world, seed=get_boxcar_constant('gaussian_floor_seed'), num_tiles=get_boxcar_constant('max_floor_tiles'))

        # We are now in progress of creating the first gen
        self.state = States.FIRST_GEN_IN_PROGRESS

        # Initialize cars randomly
        self.cars = []
        # Determine how many cars to make
        num_to_create = None
        if get_ga_constant('num_parents') - self._total_individuals_ran >= get_boxcar_constant('run_at_a_time'):
            num_to_create = get_boxcar_constant('run_at_a_time')
        else:
            num_to_create = get_ga_constant(
                'num_parents') - self._total_individuals_ran

        # @NOTE that I create the subset of cars
        for i in range(num_to_create):
            car = create_random_car(
                self.world, self.floor.winning_tile, self.floor.lowest_y, self.get_id())
            self.cars.append(car)
            

        # Add the cars to the next_pop which is used by population
        self._next_pop.extend(self.cars)

        leader = self.find_new_leader()
        self.leader = leader

        # Time to go to new state?
        if self._total_individuals_ran == get_ga_constant('num_parents'):
            self._creating_random_cars = False
            self.state = States.NEXT_GEN

    def _set_number_of_cars_alive(self) -> None:
        """
        Set the number of cars alive on the screen label
        """

        if self.state != States.REPLAY:
            total_for_gen = get_ga_constant('num_parents')
            if self.current_generation > 0:
                total_for_gen = self._next_gen_size
            num_batches = math.ceil(
                total_for_gen / get_boxcar_constant('run_at_a_time'))
            text = '{}/{} (batch {}/{})'.format(self.num_cars_alive,
                                                self.batch_size, self.current_batch, num_batches)
        else:
            text = f"<font color='red'>{'Replay' if args.replay_from_filename else 'Testing'} {np.count_nonzero(np.array([car.is_alive for car in self.cars]))}/{len(self.cars)}</font>"
        self.stats_window.current_num_alive.setText(text)

    def _set_max_fitness(self) -> None:
        """
        Sets the max fitness label
        """
        self.stats_window.best_fitness.setText(str(self.max_fitness))

    def _update(self) -> None:
        """
        Called once every 1/FPS to update everything
        """

        if not self.state == States.REPLAY and self.current_generation >= get_ga_constant("max_generations"):
            self.state = States.STOP

        if self.state == States.STOP:
            sys.exit("Max generations reached.")

        for car in self.cars:
            if not car.is_alive:
                continue
            # Did the car die/win?
            if not car.update():
                # Another individual has finished
                self._total_individuals_ran += 1
                # Decrement the number of cars alive
                self.num_cars_alive -= 1
                self._set_number_of_cars_alive()

                # If the car that just died/won was the leader, we need to find a new one
                if car == self.leader:
                    leader = self.find_new_leader()
                    self.leader = leader
                    self.game_window.leader = leader
            else:
                if not self.leader:
                    self.leader = leader
                    self.game_window.leader = leader
                else:
                    car_pos = car.position.x
                    if car_pos > self.leader.position.x:
                        self.leader = car
                        self.game_window.leader = car

        if not np.any(np.array([car.is_alive for car in self.cars])):
            if self.state == States.REPLAY and self.current_generation > 0:
                # Should we save the pop
                if args.test_from_filename:
                    self.population.individuals = self.cars

                    # Calculate fit
                    for individual in self.population.individuals:
                        individual.calculate_fitness()

                    path = "/".join(args.test_from_filename.split('/')[0:-1])

                    if not os.path.exists(path):
                        # raise Exception('{} already exists. This would overwrite everything, choose a different folder or delete it and try again'.format(path))
                        os.makedirs(path)
                    save_population(path, self.file_name, self.population, get_settings(), self.current_generation)

                self.state = States.STOP
                return

        # If the leader is valid, then just pan to the leader
        if not self.manual_control and self.leader:
            self.game_window.pan_camera_to_leader(get_boxcar_constant("should_smooth_camera_to_leader"))
        # If there is not a leader then the generation is over OR the next group of N need to run
        if not self.leader:
            # Replay state
            if self.state == States.REPLAY:
                print("REPLAY")
                cars = load_cars(
                    self.world,
                    self.floor.winning_tile,
                    self.floor.lowest_y,
                    np.inf,
                    args.replay_from_filename if args.replay_from_filename else args.test_from_filename
                )

                self.cars = cars
                self.game_window.cars = self.cars
                self.leader = self.find_new_leader()
                self.game_window.leader = self.leader
                self.current_generation += 1
                # txt = 'Replay {}/{}'.format(self.current_generation, self.num_replay_inds)
                self.stats_window.generation.setText(f"<font color='red'>{'Replay' if args.replay_from_filename else 'Testing'}</font>")
                self.stats_window.pop_size.setText(f"<font color='red'>{'Replay' if args.replay_from_filename else 'Testing'}</font>")

                self._set_number_of_cars_alive()
                return
            # Are we still in the process of just random creation?
            if self.state in (States.FIRST_GEN, States.FIRST_GEN_IN_PROGRESS):
                self._set_first_gen()
                self.game_window.leader = self.leader
                self.game_window.cars = self.cars
                self.num_cars_alive = len(self.cars)
                self.batch_size = self.num_cars_alive
                self.current_batch += 1
                self._set_number_of_cars_alive()
                return
            # Next N individuals need to run
            # We already have a population defined and we need to create N cars to run
            elif self.state == States.NEXT_GEN_CREATE_OFFSPRING:
                num_create = min(
                    self._next_gen_size - self._total_individuals_ran, get_boxcar_constant('run_at_a_time'))

                self.cars = self._create_num_offspring(num_create)
                self.batch_size = len(self.cars)
                self.num_cars_alive = len(self.cars)

                # These cars will now be part of the next pop
                self._next_pop.extend(self.cars)
                self.game_window.cars = self.cars
                leader = self.find_new_leader()
                self.leader = leader
                self.game_window.leader = leader
                # should we go to the next state?
                if (self.current_generation == 0 and (self._total_individuals_ran >= get_ga_constant('num_parents'))) or \
                        (self.current_generation > 0 and (self._total_individuals_ran >= self._next_gen_size)):
                    self.state = States.NEXT_GEN
                else:
                    self.current_batch += 1
                    self._set_number_of_cars_alive()
                return
            elif self.state in (States.NEXT_GEN, States.NEXT_GEN_COPY_PARENTS_OVER, States.NEXT_GEN_CREATE_OFFSPRING):
                self.next_generation()
                return
            else:
                raise Exception('You should not be able to get here, but if you did, awesome! Report this to me if you actually get here.')

        self.world.ClearForces()

        # Add screenshot of the image to the video
        if self.out != None:
            self.addImageToVideo()

        # Update windows
        self.game_window._update()

        # Step
        self.world.Step(1./FPS, 10, 6)

    def _crossover(self, p1_chromosome: np.ndarray, p2_chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parent chromosomes and return TWO child chromosomes
        """
        # SBX
        if random.random() <= get_ga_constant('crossover_probability'):
            return SPBX(p1_chromosome, p2_chromosome)
        else:
            return p1_chromosome, p2_chromosome

    def _mutation(self, chromosome: np.ndarray) -> None:
        """
        Randomly decide if we should perform mutation on a gene within the chromosome. This is done in place
        """
        # Gaussian
        mutation_rate = get_ga_constant('mutation_rate')
        if get_ga_constant('mutation_rate_type').lower() == 'dynamic':
            mutation_rate = mutation_rate / \
                math.sqrt(self.current_generation + 1)
        gaussian_mutation(chromosome, mutation_rate, scale=get_ga_constant('gaussian_mutation_scale'))

    def addImageToVideo(self):
        pixmap = QPixmap(self.width, self.height)
        self.render(pixmap)
        qimage = pixmap.toImage()
        buffer = qimage.bits().asstring(qimage.width() * qimage.height() * qimage.depth() // 8)
        qimage_array = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), qimage.depth() // 8))
        mat = cv2.cvtColor(qimage_array, cv2.COLOR_RGBA2RGB)
        self.out.write(mat)

    def keyPressEvent(self, event):
        global scale, default_scale
        key = event.key()
        # Zoom in
        if key == Qt.Key_C:
            scale += 1
        # Zoom out
        elif key == Qt.Key_Z:
            scale -= 1
            scale = max(scale, 1)
        elif key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self.manual_control = True
            if key == Qt.Key_W:
                direction = 'u'
            elif key == Qt.Key_A:
                direction = 'l'
            elif key == Qt.Key_S:
                direction = 'd'
            elif key == Qt.Key_D:
                direction = 'r'
            self.game_window.pan_camera_in_direction(direction, 5)
        # Reset to normal control
        elif key == Qt.Key_R:
            self.manual_control = False
        elif key == Qt.Key_E:
            scale = default_scale

    def closeEvent(self, event):
        global args
        if args.save_pop_on_close:
            save_population(args.save_pop_on_close, self.file_name, self.population, get_settings(), self.current_generation, self.run)

    def getVideo(self):
        return self.out


def save_population(population_folder: str, file_name: str, population: Population, settings_dict: Dict[str, Any], current_generation: int, datetime: str, run: str = 0) -> None:
    """
    Saves all cars in the population
    """
    # @NOTE: self.population.individuals is not the same as self.cars
    # self.cars are the cars that run at a given time for the BATCH
    # self.population.individuals is the ENTIRE population of chromosomes.
    # This will not save anything the first generation since those are just random cars and nothing has
    # been added to the population yet.

    settings_fname = os.path.join(population_folder, f'settings_{datetime}_run{run}.csv')
    pd.DataFrame(settings_dict).to_csv(settings_fname)

    settings_fname = os.path.join(population_folder, f'settings_{datetime}_run{run}.pkl')
    with open(settings_fname, 'wb') as out:
        pickle.dump(settings_dict, out)

    if file_name not in os.listdir(population_folder):
        with open(os.path.join(population_folder, file_name), "w") as population_file:
            population_file.write(get_boxcar_constant("population_headers"))

    for i, car in enumerate(population.individuals):
        car_name = f'car_{car.id}'
        print('saving {} to {}'.format(car_name, population_folder))
        save_car(
            population_folder=population_folder,
            file_name=file_name,
            car=car,
            current_generation=current_generation,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='PyGenoCar V1.0')
    # Save
    parser.add_argument('--save-best', dest='save_best', type=str,
                        help='destination folder to save best individiuals after each gen')
    parser.add_argument('--save-pop', dest='save_pop', type=str,
                        help='destination folder to save population after each gen')
    parser.add_argument('--save-pop-on-close', dest='save_pop_on_close',
                        type=str, help='destination to save the population when program exits')

    # Video
    parser.add_argument('--save-video', dest='save_video',
                        type=str, help='name of video to save')

    # Replay @NOTE: Only supports replaying the best individual. Not a list of populations.
    # parser.add_argument('--replay-from-folder', dest='replay_from_folder',
    #                     type=str, help='destination to replay individuals from')

    parser.add_argument('--replay-from-filename', dest='replay_from_filename',
                        type=str, help='destination to replay run from')

    parser.add_argument('--test-from-filename', dest='test_from_filename',
                        type=str, help='destination to test run from')

    parser.add_argument('--run', dest='run',
                        type=str, help='Index of the run')

    parser.add_argument('--seed', dest='seed',
                        type=str, help='Floor seed')

    args = parser.parse_args()
    return args


def release(win):
    win.getVideo().release()
    sys.exit("Released")


if __name__ == "__main__":
    global args
    args = parse_args()
    replay = False

    _datetime = (datetime.now()).strftime("%Y%m%d_%H%M")

    # if args.replay_from_folder:
    #     if 'settings.pkl' not in os.listdir(args.replay_from_folder):
    #         raise Exception('settings.pkl not found within {}'.format(
    #             args.replay_from_folder))
    #     settings_path = os.path.join(args.replay_from_folder, 'settings.pkl')
    #     with open(settings_path, 'rb') as f:
    #         settings.settings = pickle.load(f)
    #     replay = True

    if args.replay_from_filename or args.test_from_filename:
        filename = args.replay_from_filename if args.replay_from_filename else args.test_from_filename
        name = filename.split('/')
        replay_settings_fname = os.path.join("/".join(name[0:-1]), "settings_" + name[-1].split('.')[0] + ".pkl")

        with open(replay_settings_fname, "rb") as f:
            settings.settings = pickle.load(f)
        replay = True

        if args.test_from_filename:
            print("Test from filename:", args.test_from_filename)

            _datetime = "test_" + name[-1].split('.')[0]
            settings.update_settings_value(
                "boxcar",
                "gaussian_floor_seed",
                (random.randint(1, 1000), int),
                -1,
                "/".join(name[0:-1]),
                "settings_update_" + name[-1].split('.')[0] + ".csv",
                should_log=True
            )

    if args.save_video:
        output_file = "video_" + _datetime + ".mp4"
        output_path = os.path.join(args.save_video, output_file)
        if not os.path.exists(args.save_video):
            # raise Exception('{} already exists. This would overwrite everything, choose a different folder or delete it and try again'.format(path))
            try:
                os.makedirs(args.save_video)
            except FileExistsError:
                print("File already created")
    else:
        output_path = None

    App = QApplication(sys.argv)

    if args.run:
        print("RUN:", args.run)

        if args.seed:
            print("Seed:", args.seed)

            settings.update_settings_value(
                "boxcar",
                "gaussian_floor_seed",
                (int(args.seed), int),
                -1,
                # "/".join(name[0:-1]),
                # "settings_update_" + name[-1].split('.')[0] + "_run" + args.run + ".csv",
                should_log=False
            )

        world = b2World(get_boxcar_constant('gravity'))
        window = MainWindow(world, output_path, _datetime, replay, run=str(args.run))
        window.move(0, 0)
        window.setFixedWidth(get_window_constant('width'))
        window.setFixedHeight(get_window_constant('height'))

    else:
        stacked_widget = QStackedWidget()
        first_window = InitWindow(stacked_widget, output_path, _datetime, replay)
        stacked_widget.addWidget(first_window)
        stacked_widget.setCurrentWidget(first_window)

        stacked_widget.show()

    if args.save_video:
        App.exec_()
        try:
            atexit.register(release(window.currentWidget()))
        except:
            atexit.register(release(stacked_widget.currentWidget()))
    else:
        sys.exit(App.exec_())