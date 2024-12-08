#!usr/bin/env python3

import torch
import os
import time 
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from src.env_real_without_goal import *
from dqn_agent import *
# from data_logger import *

exp_path = os.path.dirname(os.path.realpath(__file__))
exp_path = exp_path.replace('dqn_gazebo/nodes', 'dqn_gazebo/Experimental_result')

def TestModel():
    mode = "test"
    load_episodes = 2040
    rospy.init_node('dqn_test_node_irl')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    # logger = DataLogger()
    
    NUMBER_TRIALS = 20
    TRIAL_LENGTH = 5
    state_size = 21
    action_size = 5
    
    # define the environment
    env = Env(action_size)

    # define the agent
    agent = ReinforceAgent(state_size, action_size, mode, load_episodes)
    rewards_per_trial, episodes, reward_list = [], [], []
    global_steps = 0
    
    # Init log files
    log_sim_info = open(exp_path+'/LogInfo.txt','w+')
    
    # Date / Time
    start_time = time.time()
    now_start = datetime.now()
    dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")
    
    # set goal for robot
    X_GOAL, Y_GOAL, THETA_GOAL = env.setGoalPosition()

    # Log date to files
    text = '\r\n' + '********************************************************\n'
    text = text + 'DEPLOYMENT START ==> ' + dt_string_start + '\r\n'
    text = text + '********************************************************\n'
    print(text)
    log_sim_info.write(text)
    
    for e in range(1, NUMBER_TRIALS):
        text = '\r\n' + '_____ TRIAL: ' + str(e) + ' _____' + '\r\n'
        text = text + '-----------------------------------------------------------\n'
        print(text)
        done = False
        state = env.reset()
        score = 0
        
        episode_is_done = False
        while not episode_is_done:
            state = np.float32(state)
            print(state.shape)
            # get action
            action = agent.getAction(state)
            
            # take action and return next_state, reward and other status
            next_state, reward, done, counters = env.step(action, X_GOAL, Y_GOAL, THETA_GOAL)
            next_state = np.float32(next_state)
            
            score += reward
            reward_list.append(reward)
            
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
                if counters >= TRIAL_LENGTH:
                    print("Deployment Terminated Successfully!!!")
                    text = '\r\n'+'Trial: %d | Deployment Terminated Successfully!!! \r\n'%(e)
                    text = text + '-----------------------------------------------------------\r\n'
                    log_sim_info.write('\r\n'+text)
                    
                    episode_is_done = True
                    # logger.save_data(e)
            
            # check if collision or terminated status      
            if done:
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
                np.savetxt(exp_path + '/rewards_per_trial.csv', rewards_per_trial, delimiter = ' , ')
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