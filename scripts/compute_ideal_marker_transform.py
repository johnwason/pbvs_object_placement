import rospy
import cv2
import sys
import general_robotics_toolbox as rox
from sensor_msgs.msg import CameraInfo
import numpy as np
from general_robotics_toolbox import ros_msg as rox_msg
from pbvs_object_placement_compute_ideal import get_all_payload_markers, get_aruco_dictionary, \
    get_aruco_gridboard, detect_marker, get_camera_info

def main():

    rospy.init_node("compute_ideal_marker_transform", anonymous=True)
    
    fname = sys.argv[1]
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    fixed_marker_name = sys.argv[2] # for tip panel use: python compute_ideal_marker_transform.py IMAGE_FILE leeward_mid_panel_marker_1 leeward_tip_panel_marker_2 /gripper_camera_2/camera_info
    payload_marker_name = sys.argv[3] # for mid panel use: python compute_ideal_marker_transform.py IMAGE_FILE panel_nest_marker_1 leeward_mid_panel_marker_2 /gripper_camera_2/camera_info
    camera_info_topic = sys.argv[4]
        
    print fixed_marker_name
    print payload_marker_name
    
    markers = get_all_payload_markers()
    
    fixed_marker = markers[fixed_marker_name]
    payload_marker = markers[payload_marker_name]
    
    print fixed_marker
    print payload_marker
    
    aruco_dict = get_aruco_dictionary(fixed_marker)
    camera_info = get_camera_info(camera_info_topic)
    print camera_info,aruco_dict
    
    fixed_marker_transform = detect_marker(img, fixed_marker,camera_info, aruco_dict)
    payload_marker_transform = detect_marker(img, payload_marker, camera_info, aruco_dict)
    
    print fixed_marker_transform
    print payload_marker_transform
    
    tag_to_tag_transform =  fixed_marker_transform.inv()*payload_marker_transform
    tag_to_tag_transform.parent_frame_id = fixed_marker_name
    tag_to_tag_transform.child_frame_id = payload_marker_name
    print tag_to_tag_transform
    
    ros_tf = rox_msg.transform2transform_stamped_msg(tag_to_tag_transform)
    print ros_tf
    


if __name__ == "__main__":
    main()



