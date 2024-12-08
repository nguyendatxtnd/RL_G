#!usr/bin/env python3
import csv
import torch
import rospy
import numpy as np
import os
import math
import time 
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Point, Pose
from src.dueling_dqn_env import Env
from dueling_dqn_agent import *
# from data_logger import *

exp_path = os.path.dirname(os.path.realpath(__file__))
exp_path = exp_path.replace('dueling_dqn_gazebo/nodes', 'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1')

def TestModel():
    mode = "test"
    load_episodes = 840
    rospy.init_node('dqn_test_node_irl')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    # logger = DataLogger()
    
    NUMBER_TRIALS = 500
    TRIAL_LENGTH = 5
    state_size = 22
    action_size = 5
    
    # define the environment
    env = Env(action_size)

    # define the agent
    agent = DuelingQAgent(state_size, action_size, mode, load_episodes)
    rewards_per_trial, episodes, reward_list,list_toa_do = [], [], [],[]
    global_steps = 0
    
    # Init log files
    log_sim_info = open(exp_path+'/LogInfo.txt','w+')
    
    # Date / Time
    start_time = time.time()

    now_start = datetime.now()
    dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")
    #X_GOAL, Y_GOAL, THETA_GOAL = env.setGoalPosition()
    # Log date to files
    text = '\r\n' + '********************************************************\n'
    text = text + 'DEPLOYMENT START ==> ' + dt_string_start + '\r\n'
    text = text + '********************************************************\n'
    print(text)
    i=0
    ty = 0
    csv_file_path = '/home/tung/Downloads/dueling_dqn_gazebo/save_model/csv/'
    log_sim_info.write(text)
    goal_z=[[5,5,0],[1,-3,0],[3,0,1.57]]
    
    total_dist=0
    for e in range(1, NUMBER_TRIALS):
        print("start")
        print(time.time())
        text = '\r\n' + '_____ TRIAL: ' + str(e) + ' _____' + '\r\n'
        text = text + '-----------------------------------------------------------\n'
        print(text)
        done = False
        state = env.reset()
        print('1')
        score = 0
            
        episode_is_done = False
        x_re=-2.3
        y_re=-1
        while not done:
            state = np.float32(state)
            
            # get action
            action = agent.getAction(state)
            X_GOAL, Y_GOAL, THETA_GOAL=goal_z[i]
                
            # take action and return next_state, reward and other status
            next_state, reward, done, counters,toa_do = env.step(action, X_GOAL, Y_GOAL, THETA_GOAL)
            next_state = np.float32(next_state)
            
            score += reward
            reward_list.append(reward)
            list_toa_do.append(toa_do)

            x_pre=toa_do[0]
            y_pre=toa_do[1]
            dist=math.sqrt((x_pre-x_re)**2+(y_pre-y_re)**2)
            total_dist=dist+total_dist
            
            x_re=x_pre
            y_re=y_pre

            # update state, publish actions
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
                
            # store states into log data
            # logger.store_data()
            
            # check if goal is reached
            if env.get_goal:
                env.pub_cmd_vel.publish(Twist())
                rospy.loginfo("Trial: %d | Goal [%d] completed", e, counters)
                text = '\r\n'+'Trial: %d | Goal [%d] completed \r\n'%(e, counters)
                log_sim_info.write(text)
                    
                env.get_goal = False
                    
                    
                i=i+1
                ty = ty+1
                if i==3:
                    
                    i=0
                    print("end")
                    print(total_dist)
                    print(time.time())
                    with open(csv_file_path+str(ty)+'output.csv', 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        print('da luu')
                        # Ghi từng hàng của mảng vào file CSV
                        for row in list_toa_do:
                            csv_writer.writerow(row)     
                    break
                env.get_pose_tung(X_GOAL,Y_GOAL,THETA_GOAL)
                    
                if counters >= TRIAL_LENGTH:
                    print("Deployment Terminated Successfully!!!")
                    text = '\r\n'+'Trial: %d | Deployment Terminated Successfully!!! \r\n'%(e)
                    text = text + '-----------------------------------------------------------\r\n'
                    log_sim_info.write('\r\n'+text)
                    EOFErrorpisode_is_done = True
                        # logger.save_data(e)
                    
                # check if collision or terminated status      
            if done:
                
                list_toa_do = []
                i=0
                # logger.save_data(e, done="failed")
                env.pub_cmd_vel.publish(Twist())
                result.data = [score, action]
                pub_result.publish(result)
                agent.updateTargetModel()
                rewards_per_trial.append(score)
                episodes.append(e)
                
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                rospy.loginfo('Failed, at Trial %d | Time: %d:%02d:%02d',e, h, m, s)
                text = '\r\n'+'Failed, at Trial %d | Time: %d:%02d:%02d \r\n'%(e, h, m, s)
                text = text + '-----------------------------------------------------------\r\n'
                log_sim_info.write('\r\n'+text)
                
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                #np.savetvnxt(exp_path + '/rewards_per_trial.csv', rewards_per_trial, delimiter = ' , ')
                break    
                
            global_steps += 1
                
        # Close the log file
    log_sim_info.close()
            

if __name__ == '__main__':
    try:
        TestModel()
    
    except rospy.ROSInterruptException:
        print("<--------- Test mode completed --------->")
        print('Deployment Break!')
        pass