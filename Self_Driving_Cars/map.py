# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn                  # Creating Dqn class Deep Q Network

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0          # Last x cooridnate of car
last_y = 0           # Last y cooridnate of car 
n_points = 0         # points cooridnate of car
length = 0  

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)                # 5 denotes State 5 dimension ,  3 no. of action left , right , straight , 0.9 is gamma value
action2rotation = [0,20,-20]        # first index 0 (val =0) go staight , second index 1 (val 20) rotate 20 degree clockwise(right) , index 2 rotate -20 degree
last_reward = 0                     # reward is negative if it goes to sand 
scores = []                         # store rewards

# Initializing the map
first_update = True
def init():
    global sand                  #  if sand  val 1 (pixel)
    global goal_x                # final destination upper left corner
    global goal_y               # y coordinate of final destination
    global first_update           
    sand = np.zeros((longueur,largeur))
    goal_x = 20             # initialization 
    goal_y = largeur - 20   # intiialization to not zero because we want not to collide with wall
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)              # angle of car
    rotation = NumericProperty(0)           # rotation in car
    velocity_x = NumericProperty(0)         # x coordinate velocity 
    velocity_y = NumericProperty(0)         # y coordinate velocity of car 
    velocity = ReferenceListProperty(velocity_x, velocity_y)    
    sensor1_x = NumericProperty(0)           # sensor 1 detect any sand in front of car
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)          # sensor 2 detect any sand in left of car
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)          # sensor 3  detect any sand in right of car
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)            # signal 1 recieve signal from sensor 1 (density)
    signal2 = NumericProperty(0)            # signal 2 recieve signal from sensor 2 (density)
    signal3 = NumericProperty(0)            # signal 3 recieve signal from sensor 3 (density)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos        # updation of pos of car with velocity of car
        self.rotation = rotation                            # action to rotation
        self.angle = self.angle + self.rotation             # angle of rotation of car (angle is between x axis and dir of car)
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos      # update sensors of car vector(30 , 0) as 30 is dis b/w sensor and car 
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.         # update signal of car as signal is of 20 by 20 pixel of square 
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.         # divide by 400 to get density
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:           # sensor is colliding with wall (left , right , lower , upper) signal is 1 (density is 1) very high sand  
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)              # create a car object
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]     # orientation of car with respect to goal and stablise of car
        action = brain.update(last_reward, last_signal)             #update action by getting signals  from three signals
        scores.append(brain.score())                                # adding score
        rotation = action2rotation[action]                  # update rotation
        self.car.move(rotation)                         # move car
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)     # cal mean sq. error 
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
                                                                # when car is on to some sand
        if sand[int(self.car.x),int(self.car.y)] > 0:                   
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)         # slow down the car
            last_reward = -1                                                # get reward -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)         # there is no sand move as usual
            last_reward = -0.2                                              # get reward -.02
            if distance < last_distance:
                last_reward = 0.1                                               # if car move towards the goal

        if self.car.x < 10:             
            self.car.x = 10
            last_reward = -1                                            # if car is move towards wall as condition of colliding
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10                                # 4 condition as left  , right , up , down
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:                              # update the goal if car reaches the goal
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):                    # paint the road , car , etc.....

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
