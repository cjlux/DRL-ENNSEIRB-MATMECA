###########################################################################
# Copyright 2022 Jean-Luc CHARLES
# Created: 2022-07-29
# version: 1.0 
# License: GNU GPL-3.0-or-later
###########################################################################

import pybullet as p
import pybullet_data
import numpy as np
import os, time
from math import pi

from utils.tools import is_close_to, move_to, display_joint_properties, welcome, display_link_properties
                
# Launch PyBullet server with the GUI:
pc = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

p.resetSimulation()
# set some important parameters:
p.setGravity(0,0,-9.81) # m/s^2
p.setTimeStep(0.01)     # sec

# Load the ground
planeId = p.loadURDF("plane.urdf")
    
# Load the URDF file of the robot:
startPos = [0, 0, 0.01]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
botId  = p.loadURDF("./urdf/RoboticArm_2DOF_2.urdf", startPos, startOrientation)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
#p.getCameraImage(64, 64, renderer=p.ER_BULLET_HARDWARE_OPENGL)

display_joint_properties(botId)

display_link_properties(botId)

welcome()

# Now, show some useful positions:
move_to(botId, (1,2), (0, 0))
move_to(botId, (1,2), (pi/2, 0))
move_to(botId, (1,2), (pi/2, pi/6))
move_to(botId, (1,2), (pi/2+pi/6, pi/6))
move_to(botId, (1,2), (pi/2, pi/2))
move_to(botId, (1,2), (pi/2, -pi/2))

print("Now the free run...")
move_to(botId, (1,2), (pi/2-pi/6, -pi/6), wait="Press ENTER to start, [Q] in the pybullet window to end...")
 
# disable the motor control motion for the 2 revolute joints:
p.setJointMotorControl2(botId, 1, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(botId, 2, controlMode=p.VELOCITY_CONTROL, force=0)

# now let the gravity do thejob:
step = 0
while True:
    p.stepSimulation()
    keys = p.getKeyboardEvents(physicsClientId=pc)
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        break
    time.sleep(0.01)
    
    # force the simulation to end, even if you haven't press 
    # the [Q] key in the simulation window...
    step += 1
    if step >= 200: break
        
p.disconnect(physicsClientId=pc)

