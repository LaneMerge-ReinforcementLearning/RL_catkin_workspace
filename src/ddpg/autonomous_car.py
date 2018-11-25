#!/usr/bin/python

import numpy as np
import math
import rospy

from simulator_msgs.msg import EgoVehicle, TrafficVehicles, Vehicle

class Car(object):

    action_bound = [-0.3, 0.3]
    action_dim = 2
    state_dim = 30
    dt = 0.1

    def __init__(self):

        rospy.init_node("ddpg")

        self.prev_ego_state = None

        self.goal_state = [88.57, 29.24, 0, 0, 0, 0, 2, 6, 0, 0]
        self.vel_max = 50
        self.abs_accel_max = 4
        self.steering_max = 40
        self.lr = 1.4
        self.lf = 1.4
        self.timestep = 0.02
        self.vehicle_length = 4.78
        self.vehicle_width = 2.1
        self.wheelbase = 2.8

        self.setupPublishers()

    def initializeTimings(self):

        self.prev_step_time = rospy.get_time()

    def setupPublishers(self):

        self.ego_pub = rospy.Publisher("/ego_veh_state", EgoVehicle, queue_size=1)
        self.traffic_pub = rospy.Publisher("/traffic_veh_states", TrafficVehicles, queue_size=1)

    def publishEgoState(self, state):

        ros_ego = EgoVehicle()

        ros_ego.header.stamp = rospy.Time.now()

        ros_ego.vehicle.pose.x = state[0]
        ros_ego.vehicle.pose.y = state[1]
        ros_ego.vehicle.pose.theta = state[4]

        ros_ego.vehicle.vel = state[2]
        ros_ego.vehicle.accel = state[3]
        ros_ego.vehicle.steering = state[5]

        ros_ego.vehicle.length = self.vehicle_length
        ros_ego.vehicle.width = self.vehicle_width

        self.ego_pub.publish(ros_ego)

    def publishTrafficStates(self, traffic_states):

        ros_traffic = TrafficVehicles()

        ros_traffic.header.stamp = rospy.Time.now()

        for state in traffic_states:

            vehicle = Vehicle()

            vehicle.pose.x = state[0]
            vehicle.pose.y = state[1]
            vehicle.pose.theta = state[4]

            vehicle.vel = state[2]
            vehicle.accel = state[3]
            vehicle.steering = state[5]

            vehicle.length = self.vehicle_length
            vehicle.width = self.vehicle_width

            ros_traffic.traffic.append(vehicle)

        self.traffic_pub.publish(ros_traffic)

    def step(self, state, action):

        now = rospy.get_time()
        self.timestep = now - self.prev_step_time
        self.prev_step_time = now

        state = state.reshape((3,10))

        ego_cs = state[0]
        other_cs = state[1]
        obstacle_cs = state[2]

        x_cs_ego =  ego_cs[0]
        y_cs_ego =  ego_cs[1]
        vel_cs_ego = ego_cs[2]
        accel_cs_ego = ego_cs[3]
        heading_cs_ego = ego_cs[4]
        steer_angle_cs_ego = ego_cs[5]

        steer_angle_ns_ego = steer_angle_cs_ego + math.radians(action[0])
        accel_ns_ego = accel_cs_ego + action[1]

        if steer_angle_ns_ego > math.radians(self.steering_max):
            steer_angle_ns_ego = math.radians(self.steering_max)
        elif steer_angle_ns_ego < math.radians(-self.steering_max):
            steer_angle_ns_ego = math.radians(-self.steering_max)

        if accel_ns_ego > self.abs_accel_max:
            accel_ns_ego = self.abs_accel_max

        front_wheel_pos_x = x_cs_ego + math.cos(heading_cs_ego) * (self.wheelbase / 2)
        front_wheel_pos_y = y_cs_ego + math.sin(heading_cs_ego) * (self.wheelbase / 2)

        rear_wheel_pos_x = x_cs_ego - math.cos(heading_cs_ego) * (self.wheelbase / 2)
        rear_wheel_pos_y = y_cs_ego - math.sin(heading_cs_ego) * (self.wheelbase / 2)

        front_wheel_pos_x += self.timestep * vel_cs_ego * math.cos(heading_cs_ego + steer_angle_cs_ego)
        front_wheel_pos_y += self.timestep * vel_cs_ego * math.sin(heading_cs_ego + steer_angle_cs_ego)

        rear_wheel_pos_x += self.timestep * vel_cs_ego * math.cos(heading_cs_ego)
        rear_wheel_pos_y += self.timestep * vel_cs_ego * math.sin(heading_cs_ego)

        x_ns_ego = (front_wheel_pos_x + rear_wheel_pos_x) / 2
        y_ns_ego = (front_wheel_pos_y + rear_wheel_pos_y) / 2

        heading_ns_ego = math.atan2(front_wheel_pos_y - rear_wheel_pos_y, front_wheel_pos_x - rear_wheel_pos_x)

        vel_ns_ego = vel_cs_ego + self.timestep * accel_cs_ego

        distance_left_road_ns = 8 - y_ns_ego
        distance_right_road_ns = y_ns_ego - 0

        distance_goal_x_ns = np.absolute(self.goal_state[0] - x_ns_ego)
        distance_goal_y_ns = np.absolute(self.goal_state[1] - y_ns_ego)

        if vel_ns_ego > self.vel_max:
            vel_ns_ego = self.vel_max

        ego_ns = [x_ns_ego, y_ns_ego, vel_ns_ego, accel_ns_ego, heading_ns_ego, steer_angle_ns_ego, distance_left_road_ns, distance_right_road_ns, distance_goal_x_ns, distance_goal_y_ns]

        other_ns = self.other_vehicle_step(other_cs)

        obstacle_ns = obstacle_cs

        next_state = np.array([ego_ns, other_ns, obstacle_ns])
        next_state = next_state.flatten()

        reward, done, reason = self.reward_function(ego_ns, other_ns, obstacle_ns)

        # Publish information for simulation
        self.publishEgoState(ego_ns)
        self.publishTrafficStates([other_ns, obstacle_ns])

        return next_state, reward, done, reason

    def other_vehicle_step(self, other_cs):

        other_ns = other_cs

        if 19 <= other_cs[0] <= 23:
            other_ns[3] = np.random.choice([-0.5,2.0])

        elif 30<= other_cs[0] <= 31:
            other_ns[3] = 0       

        other_ns[2] = other_cs[2] + other_cs[3] * self.timestep
        other_ns[0] = other_cs[0] + other_cs[2] * self.timestep

        return other_ns
    
    def reward_function(self, ego_ns, other_ns, obstacle_ns):

        reward = 0
        done = False
        reason = "None"

        distance_x_veh = np.absolute(ego_ns[0] - other_ns[0])
        distance_y_veh = np.absolute(ego_ns[1] - other_ns[1])

        distance_x_obs = np.absolute(ego_ns[0] - obstacle_ns[0])
        distance_y_obs = np.absolute(ego_ns[1] - obstacle_ns[1])

        distance_x_goal = ego_ns[8]
        distance_y_goal = ego_ns[9]

        # check moving backwards

        if self.prev_ego_state is not None:

            if ego_ns[0] < self.prev_ego_state[0]:
                reward = reward - 5
                reason = "GOING REVERSE"
                done = True
            else:
                reward = reward + 0.05

        self.prev_ego_state = ego_ns

        #collision_zone

        if distance_x_veh <= self.vehicle_length and distance_y_veh <= self.vehicle_width:
            reward = reward-10
            reason = "COLLIDED MOVING VEHICLE"
            done = True

        if distance_x_obs <= self.vehicle_length and distance_y_obs <= self.vehicle_width:
            reward = reward-10
            reason = "COLLIDED STOPPED VEHICLE"
            done = True
        
        #safe_zone_cost

        # if distance_x_veh <= self.vehicle_length + 1.5 and distance_y_veh < self.vehicle_width + 0.5:
        #     reward = reward-0.5
        #     #print("CLOSE TO VEHICLE"),

        #stays_on_road

        if not ((0< ego_ns[0]< 89) and (23.00 < ego_ns[1] and ego_ns[1] < 31.32)):
            reward = reward-20
            reason = "OUT OF THE ROAD"
            done = True
        
        #goal-reaching 

        if distance_x_goal < 1 and distance_y_goal < 0.5:
            reward = reward + 100
            reason = "GOALLLLLLLLLLLLLLLLLLLLLLLLLLL REACHED"
            done  = True

        #lane_changing

        if 5.5 <= ego_ns[1] <= 6.5:
            reward = reward + 0.1

        return reward, done, reason

    
    def reset(self):

        # [x, y, vel, accel, heading, steer_angle, distance_left_road, distance_right_road, distance_goal_x, distance_goal_y]

        ego_vehicle = np.array([6.326, 25.024, 18, 0, 0, 0, 6.3255, 2.108, 82.244, 4.217])
        self.prev_ego_state = ego_vehicle

        other_vehicle = np.array([3.163, 29.312, 20, 0, 0, 0, 0, 0, 0, 0])

        obstacle = np.array([61.718, 25.024, 0, 0, 0, 0, 0, 0, 0, 0])

        init_state = np.array([ego_vehicle, other_vehicle, obstacle])

        init_state = init_state.flatten()
        
        return init_state
     

    def set_fps(self, fps):
        pass

    def render(self):
        pass