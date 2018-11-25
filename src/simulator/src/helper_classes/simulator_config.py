#!/usr/bin/python

class SimulatorConfig:

    def __init__(self):

        self.ego_veh_state_topic        = "/ego_veh_state"
        self.traffic_states_topic       = "/traffic_veh_states"

        self.window_width               = 1280
        self.window_height              = 720

        self.pixels_per_meter           = 14.2259

        self.env_yaml_file              = "resources/environments/two_lane_straight_two_cars.yaml"
        self.env_bg_image_file          = "resources/environments/two_lane_straight.png"

        self.ego_veh_image_file         = "resources/ego_veh_small.png"
        self.traffic_veh_image_file     = "resources/traffic_veh_small.png"

        self.display_update_duration_s  = 0.02
