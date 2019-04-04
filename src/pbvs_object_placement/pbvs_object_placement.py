'''
Created on Apr 1, 2019

@author: wasonj
'''

import rospy
import genpy
import sys
import yaml
from geometry_msgs.msg import TransformStamped
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

def main():
    rospy.init_node("pbvs_object_placement")
    
    urdf_xml_string=rospy.get_param("robot_description")
    srdf_xml_string=rospy.get_param("robot_description_semantic")
    controller_commander=ControllerCommander()
    
    planner = Planner(controller_commander, urdf_xml_string, srdf_xml_string)
    
    transform_fname = sys.argv[1]
    camera_image_topic = sys.argv[2]
    camera_trigger_topic = sys.argv[3]
    camera_info_topic = sys.argv[4]
    
    tf_listener=PayloadTransformListener()
    
    desired_transform_msg=TransformStamped()
        
    with open(transform_fname,'r') as f:
        transform_yaml = yaml.load(f)
        
    genpy.message.fill_message_args(desired_transform_msg, transform_yaml)
    desired_transform = rox_msg.msg2transform(desired_transform_msg)
    
    markers = get_all_payload_markers()
    
    fixed_marker = markers[desired_transform.parent_frame_id]
    payload_marker = markers[desired_transform.child_frame_id]
    
    aruco_dict = get_aruco_dictionary(fixed_marker)
    camera_info = get_camera_info(camera_info_topic)
    
    time.sleep(1)
    
    dx=np.array([10000]*6)
    #move into function tiny
    while True:
        img = receive_image(camera_image_topic, camera_trigger_topic)
        
        target_pose = compute_step_gripper_target_pose(img, fixed_marker, payload_marker, desired_transform, \
                                      camera_info, aruco_dict, np.array([0.7]*6), tf_listener)
        
        plan = planner.trajopt_plan(target_pose, json_config_name = "panel_pickup")
        controller_commander.set_controller_mode(controller_commander.MODE_HALT,1,[],[])
        controller_commander.set_controller_mode(controller_commander.MODE_AUTO_TRAJECTORY,0.7,[],[])
        controller_commander.execute_trajectory(plan)
    
        
def compute_step_gripper_target_pose(img, fixed_marker, payload_marker, desired_transform, \
                 camera_info, aruco_dict, Kp, tf_listener):
    
    fixed_marker_transform, error_transform = compute_error_transform(img, fixed_marker, payload_marker, \
                                              desired_transform, camera_info, aruco_dict)
    
    gripper_to_camera_tf=tf_listener.lookupTransform("vacuum_gripper_tool", "gripper_camera_2", rospy.Time(0))
    
    world_to_vacuum_gripper_tool_tf=tf_listener.lookupTransform("world", "vacuum_gripper_tool", rospy.Time(0))
           
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
    
    return ret
    
    

def compute_error_transform(img, fixed_marker, payload_marker, desired_transform, \
                            camera_info, aruco_dict):
    
    fixed_marker_transform = detect_marker(img, fixed_marker,camera_info, aruco_dict)
    payload_marker_transform = detect_marker(img, payload_marker, camera_info, aruco_dict)
    tag_to_tag_transform =  fixed_marker_transform.inv()*payload_marker_transform
    error_transform = tag_to_tag_transform * desired_transform.inv()
    return fixed_marker_transform, error_transform
    
    

def receive_image(image_topic, camera_trigger_topic):
    trigger=rospy.ServiceProxy(camera_trigger_topic, Trigger)
    def do_trigger():
        trigger()
    t=threading.Timer(0.05,do_trigger)
    t.start()
    ros_img = rospy.wait_for_message(image_topic, Image, timeout=1.0)    
    img1 = CvBridge().imgmsg_to_cv2(ros_img)
    return img1


def get_all_payload_markers():
    payload_array = rospy.wait_for_message("payload", PayloadArray, timeout=1)
        
    markers=dict()
    
    for p in payload_array.payloads:
        for m in p.markers:
            markers[m.name] = m.marker
            
    for p in payload_array.link_markers:
        for m in p.markers:
            markers[m.name] = m.marker
            
    return markers
    

def get_aruco_dictionary(marker):    
    
    if not hasattr(cv2.aruco, marker.dictionary):
        raise ValueError("Invalid aruco-dict value")
    aruco_dict_id=getattr(cv2.aruco, marker.dictionary)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
    return aruco_dict

def get_aruco_gridboard(marker):
    #Create grid board representing the calibration target
    
    aruco_dict = get_aruco_dictionary(marker)
    
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

def get_camera_info(camera_info_topic):
    return rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=1)


def detect_marker(img, marker, camera_info, aruco_dict):
        
    camMatrix=np.reshape(camera_info.K,(3,3))
    distCoeffs=np.array(camera_info.D)
        
    parameters =  cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementWinSize=32
    parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR        
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
    img2=cv2.aruco.drawDetectedMarkers(img, corners,ids)
    img3=cv2.resize(img2,(0,0), fx=0.25,fy=0.25)
    cv2.imshow("",img3)
    cv2.waitKey(1)
    
    board = get_aruco_gridboard(marker)
    
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
         



if __name__ == '__main__':
    main()
