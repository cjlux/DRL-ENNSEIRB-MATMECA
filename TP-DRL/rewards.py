###########################################################################
# Copyright 2022 Jean-Luc CHARLES
# Date of creation: 2022-07-29
# License         : GNU GPL-3.0-or-later
###########################################################################
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
        
    reward = 0
	
    # compute the reward....
	
    return reward

