#!/usr/bin/python

from simulator import *

if __name__ == "__main__":

    rospy.init_node("simulator")
    simulator = Simulator()

    rospy.spin()
