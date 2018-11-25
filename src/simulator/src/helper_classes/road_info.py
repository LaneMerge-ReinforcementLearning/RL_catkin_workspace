#!/usr/bin/python

from pose import Pose
import math

class RoadInfo:

    # Initializing variables with default value
    def __init__(self):

        self.num_lanes         = 1

        self.lanes             = {}     # Dictionary of "lane id" : "list of lane points"


    def getNextNearestLanePoint(self, car_pose, lane_id):

        min_dist = float('Inf')
        nearest_point = None

        lane_points = self.lanes.get(lane_id)
        for point in lane_points:

            dist_from_car = math.sqrt((car_pose.x - point.x) ** 2 + (car_pose.y - point.y) ** 2)

            if dist_from_car < min_dist:
                min_dist = dist_from_car
                nearest_point = point

        # If car has already went past that point, get the next point in the list of lane points
        if nearest_point.x < car_pose.x:
            nearest_point_index = lane_points.index( nearest_point )
            nearest_point = lane_points[nearest_point_index+1]

        return nearest_point