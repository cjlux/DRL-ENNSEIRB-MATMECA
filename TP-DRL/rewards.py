import numpy as np
from numpy.linalg import norm
from math import exp, pi

def reward(self, action):
    raise Exception("You should not use this reward. Please define reward function in the file reward.py.")
    return 1

def reward_0(self, action):
        
    q1, q1_dot, q2, q2_dot, x_tg, z_tg, x_ee, z_ee = self.state
        
    reward = 0
	
    # compute the reward....
	
    return reward

def reward_1(self, action):
        
    q1, q1_dot, q2, q2_dot, x_tg, z_tg, x_ee, z_ee = self.state
        
    dist_effect_target = norm(np.array(self.effector_pos) - np.array(self.actual_target_pos))
    
    r = 0
    if dist_effect_target <= 1.e-3:
        r += 1
    else:
        r += 1.e-3/dist_effect_target
    reward = r        
    return reward
