#!/usr/bin/env python3

import rospy
import numpy as np
import math
from math import *
from geometry_msgs.msg import *
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Env():
    def __init__(self, action_size):
        self.init_x = 0.0 #m
        self.init_y = 0.0
        self.goal_x = 0.0  #| fixed goal
        self.goal_y = 0.0   #|
        self.theta_goal = 0.0 
        self.heading = 0
        self.action_size = action_size
        self.num_scan_ranges = 20
        self.initGoal = True
        self.get_goal = False
        self.prev_distance = 0
        self.const_vel = 0.225    #0.25
        self.k_r = 2 
        self.k_alpha = 15 
        self.k_beta = -3
        self.goal_dist_thres = 0.2  #0.55
        self.goal_angle_thres = 15 #degrees
        self.current_theta = 0
        self.goal_counters = 0
        self.nearby_distance = 0.8
        self.safe_dist = 1.0
        self.lidar = []
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # self.list_goal_x = [0.5, 1.3, 2.0, 2.0, 3.2]
        # self.list_goal_y = [0.0, -1.0, 0.4, 1.5, -2.5]
        
    
    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, cur_theta = euler_from_quaternion(orientation_list)

        self.current_theta = cur_theta #radian
        
        return self.position.x, self.position.y, self.current_theta
    
    def setGoalPosition(self):
        goal_msg = rospy.wait_for_message('move_base_simple/goal', PoseStamped)
        X_GOAL = goal_msg.pose.position.x
        Y_GOAL = goal_msg.pose.position.y
        orientation_q = goal_msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        THETA_GOAL = degrees(yaw)
        
        return X_GOAL, Y_GOAL, THETA_GOAL

    def getState(self, scan):
        self.scan_range = []
        heading = self.heading
        min_range = 0.35
        done = False
        object_nearby = False
        near_goal = False
        
        # Formatting the self.scan_range to feed algorithm
        # self.scan_range.append(max(scan[8], scan[9], scan[10]))
        # self.scan_range.append(max(scan[45], scan[46], scan[47]))
        # self.scan_range.append(max(scan[82], scan[83], scan[84]))
        # self.scan_range.append(max(scan[119], scan[120], scan[121]))
        # self.scan_range.append(max(scan[156], scan[157], scan[158]))
        # self.scan_range.append(max(scan[193], scan[194], scan[195]))
        # self.scan_range.append(max(scan[230], scan[231], scan[232]))
        # self.scan_range.append(max(scan[267], scan[268], scan[269]))
        # self.scan_range.append(max(scan[304], scan[305], scan[306]))
        # self.scan_range.append(max(scan[341], scan[342], scan[343]))
        # self.scan_range.append(max(scan[378], scan[379], scan[380]))
        # self.scan_range.append(max(scan[415], scan[416], scan[417]))
        # self.scan_range.append(max(scan[452], scan[453], scan[454]))
        # self.scan_range.append(max(scan[489], scan[490], scan[491]))
        # self.scan_range.append(max(scan[526], scan[527], scan[528]))
        # self.scan_range.append(max(scan[563], scan[564], scan[565]))
        # self.scan_range.append(max(scan[600], scan[601], scan[602]))
        # self.scan_range.append(max(scan[637], scan[638], scan[639]))
        # self.scan_range.append(max(scan[674], scan[675], scan[676]))
        # self.scan_range.append(max(scan[711], scan[712], scan[713]))
        self.scan_range.append(scan.ranges[9])
        self.scan_range.append(scan.ranges[46])
        self.scan_range.append(scan.ranges[83])
        self.scan_range.append(scan.ranges[120])
        self.scan_range.append(scan.ranges[157])
        self.scan_range.append(scan.ranges[194])
        self.scan_range.append(scan.ranges[231])
        self.scan_range.append(scan.ranges[268])
        self.scan_range.append(scan.ranges[305])
        self.scan_range.append(scan.ranges[342])
        self.scan_range.append(scan.ranges[379])
        self.scan_range.append(scan.ranges[416])
        self.scan_range.append(scan.ranges[453])
        self.scan_range.append(scan.ranges[490])
        self.scan_range.append(scan.ranges[527])
        self.scan_range.append(scan.ranges[564])
        self.scan_range.append(scan.ranges[601])
        self.scan_range.append(scan.ranges[638])
        self.scan_range.append(scan.ranges[675])
        self.scan_range.append(scan.ranges[712])
        
        for i in range(len(self.scan_range)):
            if self.scan_range[i] == float('Inf'):
                self.scan_range[i] = 3.5
            elif np.isnan(self.scan_range[i]):
                self.scan_range = 0.0
            self.scan_range[i] = round(self.scan_range[i], 4)

        # Check object_nearbyness
        if min(self.scan_range) < self.nearby_distance:
            object_nearby = True
        
        # Check object collision
        if min_range > min(self.scan_range) > 0:
            done = True
        
        # Check goal is near | reached
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance <= 0.5:
            near_goal = True

        return self.scan_range + [heading], object_nearby, done, near_goal

    def setReward(self, state, done, action):
        yaw_reward = []
        heading = state[-1]

        for i in range(self.action_size):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        reward = round(yaw_reward[action] * self.action_size, 2)

        if done:
            rospy.loginfo("*****************")
            rospy.loginfo("* COLLISION !!! *")
            rospy.loginfo("*****************")
            reward = -650.
            self.pub_cmd_vel.publish(Twist())
            rospy.spin(5)

        return reward

    ###############################################################################################################
    def FeedBackControl(self, odom):
        x, y, theta = self.getOdometry(odom)
                
        if self.theta_goal >= pi:
            theta_goal_norm = self.theta_goal - 2 * pi
        else:
            theta_goal_norm = self.theta_goal
        
        ro = sqrt( pow( ( self.goal_x - x ) , 2 ) + pow( ( self.goal_y - y ) , 2) )
        lamda = atan2( self.goal_y - y , self.goal_x - x )
        
        alpha = (lamda -  theta + pi) % (2 * pi) - pi
        beta = (self.theta_goal - lamda + pi) % (2 * pi) - pi

        if ro < self.goal_dist_thres and degrees(abs(theta - theta_goal_norm)) < self.goal_angle_thres:
            rospy.loginfo("********************")
            rospy.loginfo("* GOAL REACHED !!! *")
            rospy.loginfo("********************")
            v = 0
            w = 0
            v_scal = 0
            w_scal = 0
            vel_cmd = Twist()
            vel_cmd.linear.x = v_scal
            vel_cmd.angular.z = w_scal
            self.pub_cmd_vel.publish(vel_cmd)
            self.goal_counters += 1
            self.goal_x , self.goal_y, self.theta_goal = self.setGoalPosition()
            if self.goal_counters > 5:
                self.goal_counters = 0
                
        else:
            v = self.k_r * ro
            w = self.k_alpha * alpha + self.k_beta * beta
            v_scal = v / abs(v) * self.const_vel
            w_scal = w / abs(v) * self.const_vel
            vel_cmd = Twist()
            vel_cmd.linear.x = v_scal
            vel_cmd.angular.z = w_scal
            self.pub_cmd_vel.publish(vel_cmd)
        
    ###############################################################################################################

    def step(self, action, goal_x, goal_y, theta_goal):
        data = None
        odom = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass
            
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.theta_goal = theta_goal
        state, object_nearby, done, near_goal = self.getState(data)
        
        if (not object_nearby) or near_goal:
            self.FeedBackControl(odom)
        
        else:
            max_angular_vel = 0.75  #1.5 0.5
            ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

            vel_cmd = Twist()
            vel_cmd.linear.x = self.const_vel
            vel_cmd.angular.z = ang_vel
            self.pub_cmd_vel.publish(vel_cmd)   
            reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done, self.goal_counters

    def reset(self):
        data = None
        odom = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan)
                odom = rospy.wait_for_message('/odom', Odometry)
            except:
                pass

        self.init_x, self.init_y, self.current_theta = self.getOdometry(odom)
        
        state, object_nearby, done, near_goal = self.getState(data)
        self.goal_counters = 0
        self.lidar = state

        return np.asarray(state)