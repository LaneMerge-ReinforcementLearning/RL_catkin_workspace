#!/usr/bin/python

from simulator_helper_functions import *


def updateTrafficVehiclesPositions(sim_obj, now):

    dt = now - sim_obj.last_update_time

    for vehicle in sim_obj.traffic:

        front_wheel_pos = vehicle.pose + Pose(math.cos(vehicle.pose.theta), math.sin(vehicle.pose.theta), 0).multiply(vehicle.wheel_base / 2)

        rear_wheel_pos = vehicle.pose - Pose(math.cos(vehicle.pose.theta), math.sin(vehicle.pose.theta), 0).multiply(vehicle.wheel_base / 2)

        front_wheel_pos.x += dt * vehicle.vel * math.cos(vehicle.pose.theta + math.radians(vehicle.steering))
        front_wheel_pos.y += dt * vehicle.vel * math.sin(vehicle.pose.theta + math.radians(vehicle.steering))

        rear_wheel_pos.x += dt * vehicle.vel * math.cos(vehicle.pose.theta)
        rear_wheel_pos.y += dt * vehicle.vel * math.sin(vehicle.pose.theta)

        vehicle.pose = (front_wheel_pos + rear_wheel_pos).divide(2)

        vehicle.pose.theta = math.atan2(front_wheel_pos.y - rear_wheel_pos.y, front_wheel_pos.x - rear_wheel_pos.x)

        vehicle.vel += dt * vehicle.accel

    sim_obj.last_update_time = now