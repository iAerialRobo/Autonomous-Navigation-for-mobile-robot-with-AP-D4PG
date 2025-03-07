#!/usr/bin/env python3
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PointStamped

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = '/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/scripts/model.sdf'
    
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = 2
        self.goal_position = Pose()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    # def getPosition(self, position_check=False, delete=False):
    #     if delete:
    #         self.deleteModel()

    #     if self.stage != 4:
    #         while position_check:
    #             goal_x = random.randrange(-12, 13) / 10.0
    #             goal_y = random.randrange(-12, 13) / 10.0
    #             if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
    #                 position_check = True
    #             elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
    #                 position_check = True
    #             elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
    #                 position_check = True
    #             elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
    #                 position_check = True
    #             elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
    #                 position_check = True
    #             else:
    #                 position_check = False

    #             if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
    #                 position_check = True

    #             self.goal_position.position.x = goal_x
    #             self.goal_position.position.y = goal_y

    #     else:
    #         while position_check:
    #             goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
    #             goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

    #             self.index = random.randrange(0, 13)
    #             print(self.index, self.last_index)
    #             if self.last_index == self.index:
    #                 position_check = True
    #             else:
    #                 self.last_index = self.index
    #                 position_check = False

    #             self.goal_position.position.x = goal_x_list[self.index]
    #             self.goal_position.position.y = goal_y_list[self.index]

    #     time.sleep(0.5)
    #     self.respawnModel()

    #     self.last_goal_x = self.goal_position.position.x
    #     self.last_goal_y = self.goal_position.position.y

    #     return self.goal_position.position.x, self.goal_position.position.y
    
    
    
    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 4:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                # goal_x = -1.1
                # goal_y = 1.1
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 13)
                print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]
                       
                # self.goal_position.position.x = -1.1
                # self.goal_position.position.y =1.1

        time.sleep(0.5)
        self.respawnModel()
        point_pub = rospy.Publisher('/clicked_point', PointStamped, queue_size=10)
        rate = rospy.Rate(1)  # 发布频率为1Hz

        # while not rospy.is_shutdown():
        #     # 创建一个 PointStamped 消息
        #     point = PointStamped()
        #     point.header.frame_id = "odom"  # 使用 map 帧
        #     point.header.stamp = rospy.Time.now()
        #     point.point.x = self.goal_x # 设置点的 x 坐标
        #     point.point.y = self.goal_y  # 设置点的 y 坐标
        #     point.point.z = 0.0  # 设置点的 z 坐标

        # # 发布点消息
        #     point_pub.publish(point)

        #     rate.sleep()
        
        point = PointStamped()
        point.header.frame_id = "odom"  # 使用 map 帧
        point.header.stamp = rospy.Time.now()
        point.point.x =  self.goal_position.position.x # 设置点的 x 坐标
        point.point.y =  self.goal_position.position.y  # 设置点的 y 坐标
        point.point.z = 0.0  # 设置点的 z 坐标
        point_pub.publish(point)
        rate.sleep()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
