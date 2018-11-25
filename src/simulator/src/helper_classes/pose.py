#!/usr/bin/python

class Pose:

    # Initializing variables with default value
    def __init__(self, pos_x=0, pos_y=0, theta=0):

        self.x = pos_x
        self.y = pos_y

        self.theta = theta

    def __eq__(self, other):

        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __add__(self, other):

        return Pose(self.x+other.x, self.y+other.y, self.theta+other.theta)

    def __sub__(self, other):

        return Pose(self.x-other.x, self.y-other.y, self.theta-other.theta)

    def multiply(self, factor):

        return Pose(self.x*factor, self.y*factor, self.theta*factor)

    def divide(self, factor):

        return Pose(self.x/factor, self.y/factor, self.theta/factor)

    def getPosition(self):

        return (self.x, self.y)