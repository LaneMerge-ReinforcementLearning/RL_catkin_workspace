#!/usr/bin/python

from pose import Pose


class Vehicle:

    # Initializing variables with default value
    def __init__(self, veh_id=0, veh_init_pos=(0, 0), veh_init_theta=0.0):

        self.veh_id         = veh_id

        self.wheel_base     = 2.81
        self.length         = 4.78
        self.width          = 2.1

        self.pose           = Pose(pos_x=veh_init_pos[0], pos_y=veh_init_pos[1], theta=veh_init_theta)

        self.steering       = 0.0
        self.vel            = 0.0
        self.accel          = 0.0

        self.max_steering_angle = 40.0      # degrees
        self.max_vel            = 22.352    # mps
        self.max_accel          = 4.0       # mps2

        self.traffic_veh    = False
