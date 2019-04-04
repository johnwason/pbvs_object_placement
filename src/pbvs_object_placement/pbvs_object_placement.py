'''
Created on Apr 1, 2019

@author: wasonj
'''

from __future__ import absolute_import

import rospy
import genpy
import sys
import yaml
from geometry_msgs.msg import TransformStamped, Wrench, Vector3
from sensor_msgs.msg import CameraInfo, Image
import numpy as np
import general_robotics_toolbox as rox
from general_robotics_toolbox import ros_msg as rox_msg
from industrial_payload_manager.msg import PayloadArray
import cv2
from industrial_payload_manager.payload_transform_listener import PayloadTransformListener
import time
from std_srvs.srv import Trigger
from cv_bridge import CvBridge, CvBridgeError
from arm_composites_manufacturing_process.planner import Planner
from safe_kinematic_controller.ros.commander import ControllerCommander
import threading
import Queue
from .msg import PBVSPlacementAction, PBVSPlacementGoal



def main():
    rospy.init_node("pbvs_object_placement")
    
    urdf_xml_string=rospy.get_param("robot_description")
    srdf_xml_string=rospy.get_param("robot_description_semantic")
    controller_commander=ControllerCommander()
    
        
    transform_fname = sys.argv[1]
    camera_image_topic = sys.argv[2]
    camera_trigger_topic = sys.argv[3]
    camera_info_topic = sys.argv[4]
    
    controller = PBVSPlacementController(controller_commander, urdf_xml_string, srdf_xml_string, \
                               camera_image_topic, camera_trigger_topic, camera_info_topic)
    
    desired_transform_msg=TransformStamped()
        
    with open(transform_fname,'r') as f:
        transform_yaml = yaml.load(f)
        
    genpy.message.fill_message_args(desired_transform_msg, transform_yaml)
    
    goal = PBVSPlacementGoal()
    goal.desired_transform = desired_transform_msg
    
    goal.stage1_tol_p = 0.1
    goal.stage1_tol_r = np.deg2rad(0.25)
    goal.stage2_tol_p = 0.1
    goal.stage2_tol_r = np.deg2rad(0.25)
    goal.stage3_tol_p = 0.001
    goal.stage3_tol_r = np.deg2rad(0.05)
    
    goal.stage1_kp = np.array([0.7] * 6)
    goal.stage2_kp = np.array([0.7] * 6)
    goal.stage3_kp = np.array([0.2] * 6)
    
    goal.stage2_z_offset = 0.1
    
    goal.abort_force = Wrench(Vector3(500,500,500), Vector3(100,100,100))
    goal.placement_force = Wrench(Vector3(0,0,300), Vector3(0,0,0))
    goal.force_ki = np.array([1e-6]*6)        
    
    time.sleep(1)
    
    controller.set_goal(goal)
    
    dx=np.array([10000]*6)
    
    controller.pbvs_stage1()
    
    
    
class PBVSPlacementController(object):
    
    def __init__(self, controller_commander, urdf_xml_string, srdf_xml_string,
                 camera_image_topic, camera_trigger_topic, camera_info_topic
                 ):
        self.controller_commander=controller_commander
        self.planner=Planner(controller_commander, urdf_xml_string, srdf_xml_string)
        
        self.tf_listener=PayloadTransformListener()        
        
        self.camera_info = self.get_camera_info(camera_info_topic)
        
        self.img_queue=Queue.Queue(1)
        self.camera_trigger = rospy.ServiceProxy(camera_trigger_topic, Trigger)
        self.camera_sub = rospy.Subscriber(camera_image_topic, Image, self._ros_img_cb)
    
    def _ros_img_cb(self, ros_img):
        img1 = CvBridge().imgmsg_to_cv2(ros_img)
        try:
            self.img_queue.get_nowait()
        except Queue.Empty: pass
        self.img_queue.put_nowait(img1)
    
    def set_goal(self, params_msg):
        
        self.desired_transform=rox_msg.msg2transform(params_msg.desired_transform)
        
        self.markers = self.get_all_payload_markers()
        
        self.fixed_marker = self.markers[self.desired_transform.parent_frame_id]
        self.payload_marker = self.markers[self.desired_transform.child_frame_id]
        
        self.aruco_dict = self.get_aruco_dictionary(self.fixed_marker)
        
        self.stage1_tol_p = params_msg.stage1_tol_p
        self.stage1_tol_r = params_msg.stage1_tol_r
        self.stage2_tol_p = params_msg.stage2_tol_p
        self.stage2_tol_r = params_msg.stage2_tol_r
        self.stage3_tol_p = params_msg.stage3_tol_p
        self.stage3_tol_r = params_msg.stage3_tol_r
        
        self.stage1_kp = params_msg.stage1_kp
        self.stage2_kp = params_msg.stage2_kp
        self.stage3_kp = params_msg.stage3_kp
        
        self.stage2_z_offset = params_msg.stage2_z_offset
        self.force_ki = params_msg.force_ki
        
        def unpack_wrench(w):
            return np.array([w.torque.x, w.torque.y, w.torque.z, w.force.x, w.force.y, w.force.z])
        
        self.abort_force = unpack_wrench(params_msg.abort_force)
        self.placement_force = unpack_wrench(params_msg.placement_force)
       
    def compute_step_gripper_target_pose(self, img, Kp):
        
        fixed_marker_transform, error_transform = self.compute_error_transform(img)
        
        gripper_to_camera_tf=self.tf_listener.lookupTransform("vacuum_gripper_tool", "gripper_camera_2", rospy.Time(0))
        
        world_to_vacuum_gripper_tool_tf=self.tf_listener.lookupTransform("world", "vacuum_gripper_tool", rospy.Time(0))
               
        #Scale by Kp       
        k, theta = rox.R2rot(error_transform.R)
        r = np.multiply(k*theta, Kp[0:3])
        r_norm = np.linalg.norm(r)
        if (r_norm < 1e-6):
            error_transform2_R = np.eye(3)
        else:
            error_transform2_R = rox.rot(r/r_norm, r_norm)
        error_transform2_p = np.multiply(error_transform.p,(Kp[3:6]))
         
        error_transform2 = rox.Transform(error_transform2_R, error_transform2_p)
        
        gripper_to_fixed_marker_tf = gripper_to_camera_tf*fixed_marker_transform
        gripper_to_desired_fixed_marker_tf = gripper_to_fixed_marker_tf*error_transform2
        
        #print gripper_to_fixed_marker_tf
         
        
        ret = world_to_vacuum_gripper_tool_tf * (gripper_to_desired_fixed_marker_tf * gripper_to_fixed_marker_tf.inv()).inv()
        
        #print world_to_vacuum_gripper_tool_tf
        #print ret
        
        print error_transform
        
        return ret, error_transform
    
    

    def compute_error_transform(self, img):
        
        fixed_marker_transform = self.detect_marker(img, self.fixed_marker)
        payload_marker_transform = self.detect_marker(img, self.payload_marker)
        tag_to_tag_transform =  fixed_marker_transform.inv()*payload_marker_transform
        error_transform = tag_to_tag_transform * self.desired_transform.inv()
        return fixed_marker_transform, error_transform

    def receive_image(self):
        try:
            self.img_queue.get_nowait()
        except Queue.Empty: pass
        self.camera_trigger()
        return self.img_queue.get(timeout=10.0)

    def get_all_payload_markers(self):
        payload_array = rospy.wait_for_message("payload", PayloadArray, timeout=1)
            
        markers=dict()
        
        for p in payload_array.payloads:
            for m in p.markers:
                markers[m.name] = m.marker
                
        for p in payload_array.link_markers:
            for m in p.markers:
                markers[m.name] = m.marker
                
        return markers
    

    def get_aruco_dictionary(self,marker):    
        
        if not hasattr(cv2.aruco, marker.dictionary):
            raise ValueError("Invalid aruco-dict value")
        aruco_dict_id=getattr(cv2.aruco, marker.dictionary)
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
        return aruco_dict

    def get_aruco_gridboard(self, marker):
        #Create grid board representing the calibration target
        
        aruco_dict = self.get_aruco_dictionary(marker)
        
        if isinstance(marker.dictionary,basestring):
            if not marker.dictionary.startswith('DICT_'):
                raise ValueError("Invalid aruco-dict value")
        
            
        elif isinstance(marker.dictionary,int):
            aruco_dict = cv2.aruco.Dictionary_get(marker.dictionary)
        else:
            aruco_dict_id=marker.dictionary
        board=cv2.aruco.GridBoard_create(marker.markersX, marker.markersY, \
                                         marker.markerLength, marker.markerSpacing, aruco_dict,\
                                         marker.firstMarker)
        return board

    def get_camera_info(self,camera_info_topic):
        return rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=1)


    def detect_marker(self, img, marker):
            
        camMatrix=np.reshape(self.camera_info.K,(3,3))
        distCoeffs=np.array(self.camera_info.D)
            
        parameters =  cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementWinSize=32
        parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR        
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=parameters)
        
        img2=cv2.aruco.drawDetectedMarkers(img, corners,ids)
        img3=cv2.resize(img2,(0,0), fx=0.25,fy=0.25)
        cv2.imshow("",img3)
        cv2.waitKey(1)
        
        board = self.get_aruco_gridboard(marker)
        
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs)
        
        if (retval <= 0):
            raise Exception("Invalid image")
        
        Ra, b = cv2.Rodrigues(rvec)
        a_pose=rox.Transform(Ra,tvec)
        
        frame_with_markers_and_axis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frame_with_markers_and_axis    =    cv2.aruco.drawAxis(    frame_with_markers_and_axis,  camMatrix, distCoeffs, rvec, tvec, 0.2    )
        frame_with_markers_and_axis=cv2.resize(frame_with_markers_and_axis,(0,0), fx=0.25,fy=0.25)
        cv2.imshow("transform", frame_with_markers_and_axis)
        cv2.waitKey(1)
        
        return a_pose
    
    def pbvs(self, kp, tols_p, tols_r, abort_force, max_iters = 25):
    
        i=0
        while True:
            
            if i > max_iters:
                raise Exception("Placement controller timeout")
            
            img = self.receive_image()
            
            target_pose, err = self.compute_step_gripper_target_pose(img, kp)
            
            err_p = np.linalg.norm(err.p)
            err_r = np.abs(rox.R2rot(err.R)[1]) 
            
            if err_p < tols_p and err_r < tols_r:
                return err_p, err_r
            
            plan = self.planner.trajopt_plan(target_pose, json_config_name = "panel_pickup")            
            self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY,0.7,[], abort_force)
            self.controller_commander.execute_trajectory(plan)
            i+=1
    
    def pbvs_stage1(self):
        return self.pbvs(self.stage1_kp, self.stage1_tol_p, self.stage1_tol_r, self.abort_force)
        
    def pbvs_stage2(self):
        return self.pbvs(self.stage2_kp, self.stage2_tol_p, self.stage2_tol_r, self.abort_force)
     
    def pbvs_stage3(self):
        return self.pbvs(self.stage3_kp, self.stage3_tol_p, self.stage3_tol_r, self.abort_force)
     



if __name__ == '__main__':
    main()
