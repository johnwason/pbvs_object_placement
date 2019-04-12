'''
Created on Apr 1, 2019

@author: wasonj
'''

#from __future__ import absolute_import

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
from safe_kinematic_controller.msg import ControllerState as controllerstate
import threading
import Queue
from .msg import PBVSPlacementAction, PBVSPlacementGoal
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from placement_functions import QP_abbirb6640, trapgen
from scipy.linalg import logm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy


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
    
    controller.controller_commander.set_controller_mode(controller.controller_commander.MODE_HALT,0.7,[], [])
    desired_transform_msg=TransformStamped()
        
    with open(transform_fname,'r') as f:
        transform_yaml = yaml.load(f)
        
    genpy.message.fill_message_args(desired_transform_msg, transform_yaml)
    
    goal = PBVSPlacementGoal()
    goal.desired_transform = desired_transform_msg
    
    goal.stage1_tol_p = 0.05
    goal.stage1_tol_r = np.deg2rad(1)
    goal.stage2_tol_p = 0.05
    goal.stage2_tol_r = np.deg2rad(1)
    goal.stage3_tol_p = 0.001
    goal.stage3_tol_r = np.deg2rad(0.2)
    
    goal.stage1_kp = np.array([0.90] * 6)
    goal.stage2_kp = np.array([0.90] * 6)
    goal.stage3_kp = np.array([0.5] * 6)
    
    goal.stage2_z_offset = 0.05
    
    goal.abort_force = Wrench(Vector3(500,500,500), Vector3(100,100,100))
    goal.placement_force = Wrench(Vector3(0,0,300), Vector3(0,0,0))
    goal.force_ki = np.array([1e-6]*6)        
    
    time.sleep(1)
    
    controller.set_goal(goal)
    
    dx=np.array([10000]*6)
    
    
    tic = time.time()
    rospy.loginfo("started")
    controller.pbvs_stage1()
    rospy.loginfo("finished stage 1")
    controller.pbvs_stage2()
    rospy.loginfo("finished stage 2")
    controller.pbvs_jacobian()
    rospy.loginfo("finished stage 3")
    print "Time:", time.time()-tic
    #controller.test()    
    
    
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


        self.controller_state_sub = rospy.Subscriber("controller_state", controllerstate, self.ft_cb)
        self.FTdata = None
        self.ft_flag = False
        self.FTdata_0 = self.FTdata           
        # Compliance controller parameters
        self.F_d_set1 = -120
        self.F_d_set2 = -220
        self.Kc = 0.000025
        
        self.client = actionlib.SimpleActionClient("joint_trajectory_action", FollowJointTrajectoryAction) 
        self.K_pbvs = 0.3          
 
           
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
        self.K_pbvs=self.stage3_kp
        
        self.stage2_z_offset = params_msg.stage2_z_offset
        self.force_ki = params_msg.force_ki
        
        def unpack_wrench(w):
            return np.array([w.torque.x, w.torque.y, w.torque.z, w.force.x, w.force.y, w.force.z])
        
        self.abort_force = unpack_wrench(params_msg.abort_force)
        self.placement_force = unpack_wrench(params_msg.placement_force)
        
        self._aborted=False
       
    def compute_step_gripper_target_pose(self, img, Kp, no_z = False, z_offset = 0):
        
        fixed_marker_transform, payload_marker_transform, error_transform = self.compute_error_transform(img)
        
        if no_z:
            error_transform.p[2] = 0
        else:
            error_transform.p[2] -= z_offset
        
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
        
        #print error_transform
        
        return ret, error_transform
    
    

    def compute_error_transform(self, img):
        
        fixed_marker_transform = self.detect_marker(img, self.fixed_marker)
        payload_marker_transform = self.detect_marker(img, self.payload_marker)
        tag_to_tag_transform =  fixed_marker_transform.inv()*payload_marker_transform
        error_transform = tag_to_tag_transform * self.desired_transform.inv()
        return fixed_marker_transform,payload_marker_transform, error_transform

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
            #cv2.waitKey()
            raise Exception("Invalid image")
        
        Ra, b = cv2.Rodrigues(rvec)
        a_pose=rox.Transform(Ra,tvec)
        
        frame_with_markers_and_axis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frame_with_markers_and_axis    =    cv2.aruco.drawAxis(    frame_with_markers_and_axis,  camMatrix, distCoeffs, rvec, tvec, 0.2    )
        frame_with_markers_and_axis=cv2.resize(frame_with_markers_and_axis,(0,0), fx=0.25,fy=0.25)
        cv2.imshow("transform", frame_with_markers_and_axis)
        cv2.waitKey(1)
        
        return a_pose
    
    def pbvs(self, kp, tols_p, tols_r, abort_force, max_iters = 25, no_z = False, z_offset = 0):

        i=0
        while True:
            
            if i > max_iters:
                raise Exception("Placement controller timeout")
            
            if self._aborted:
                raise Exception("Operation aborted")
            
            img = self.receive_image()
            
            target_pose, err = self.compute_step_gripper_target_pose(img, kp, no_z = no_z, z_offset = z_offset)
                        
            err_p = np.linalg.norm(err.p)
            if no_z:
                err_p = np.linalg.norm(err.p[0:2])
            err_r = np.abs(rox.R2rot(err.R)[1]) 
            
            if err_p < tols_p and err_r < tols_r:
                return err_p, err_r
            
            #target_pose.p[1]-=0.01 # Offset a little to avoid panel overlap
            plan = self.planner.trajopt_plan(target_pose, json_config_name = "panel_placement")
            
            if self._aborted:
                raise Exception("Operation aborted")            
            self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY,0.7,[], abort_force)
            self.controller_commander.execute_trajectory(plan)
            i+=1
    
    def pbvs_stage1(self):
        rospy.loginfo("stage 1 PBVS")
        return self.pbvs(self.stage1_kp, self.stage1_tol_p, self.stage1_tol_r, self.abort_force, no_z = True)
        
    def pbvs_stage2(self):
        rospy.loginfo("stage 2 PBVS")
        return self.pbvs(self.stage2_kp, self.stage2_tol_p, self.stage2_tol_r, self.abort_force, z_offset = self.stage2_z_offset)
     
    def pbvs_stage3(self):
        print "stage 3"
        return self.pbvs(self.stage3_kp, self.stage3_tol_p, self.stage3_tol_r, self.abort_force)
     
    def abort(self):
        self._aborted=True
        self.controller_commander.stop_trajectory()

    # Added by YC
    def ft_cb(self, data):
        self.FTdata = np.array([data.ft_wrench.torque.x,data.ft_wrench.torque.y,data.ft_wrench.torque.z,\
        data.ft_wrench.force.x,data.ft_wrench.force.y,data.ft_wrench.force.z])
        self.ft_flag=True
        
    
    def trapezoid_gen(self,target,current_joint_angles,acc,dcc,vmax):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        goal.trajectory.header.frame_id='/world'
    
        dist = np.array(target-current_joint_angles)
        xf = max(abs(dist))
        #print "Dist:====================", xf
        [x0,v0,a0,ta,tb,tf] = trapgen(0,xf,0,0,vmax,acc,dcc,0)
        [xa,va,aa,ta,tb,tf] = trapgen(0,xf,0,0,vmax,acc,dcc,ta)
        [xb,vb,ab,ta,tb,tf] = trapgen(0,xf,0,0,vmax,acc,dcc,tb)
    		
        p1=JointTrajectoryPoint()
        p1.positions = current_joint_angles
        p1.time_from_start = rospy.Duration(0)
        
        p2=JointTrajectoryPoint()
        p2.positions = np.array(p1.positions) + dist*xa
        p2.time_from_start = rospy.Duration(ta)
        
        p3=JointTrajectoryPoint()
        p3.positions = np.array(p1.positions) + dist*xb
        p3.time_from_start = rospy.Duration(tb)
    
        p4=JointTrajectoryPoint()
        p4.positions = target
        p4.velocities = np.zeros((6,))
        p4.accelerations = np.zeros((6,))
        p4.time_from_start = rospy.Duration(tf)
        
        goal.trajectory.points.append(p1)
        goal.trajectory.points.append(p2)
        goal.trajectory.points.append(p3)
        goal.trajectory.points.append(p4)
        
        return goal
     
    def test(self):
        img = cv2.imread('12_57_46_523.png')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fixed_marker_transform, payload_marker_transform, error_transform = self.compute_error_transform(img)  
        tag_to_tag_transform =  fixed_marker_transform.inv()*payload_marker_transform
        print "Test1:",fixed_marker_transform, payload_marker_transform
        print "Test2:",tag_to_tag_transform
        print "Test3:",error_transform
        
    def pbvs_jacobian(self):
        rospy.loginfo("stage 3 PBVS")
        self.controller_commander.set_controller_mode(self.controller_commander.MODE_AUTO_TRAJECTORY, 0.7, [], [])
        tvec_err = [100,100,100]
        rvec_err = [100,100,100]
        
        self.FTdata_0 = self.FTdata
        
        error_transform= rox.Transform(rox.rpy2R([2,2,2]), np.array([100,100,100]))
        
        FT_data_ori = []
        FT_data_biased = []
        err_data_p = []
        err_data_rpy = []
        joint_data = []
        time_data = []
        #TODO: should listen to stage_3_tol_r not 1 degree
        
        while(error_transform.p[2]>0.01 or np.linalg.norm([error_transform.p[0],error_transform.p[1]]) > self.stage3_tol_p or np.linalg.norm(rox.R2rpy(error_transform.R)) > self.stage3_tol_r):
            
            img = self.receive_image()

            fixed_marker_transform, payload_marker_transform, error_transform = self.compute_error_transform(img)  
            #print self.desired_transform.R.T, -fixed_marker_transform.R.dot(self.desired_transform.p)

            R_desired_cam = fixed_marker_transform.R.dot(self.desired_transform.R)
            t_desired_cam = -fixed_marker_transform.R.dot(self.desired_transform.p)
            
            # Compute error directly in the camera frame
            k, theta = rox.R2rot(np.dot(payload_marker_transform.R , R_desired_cam.transpose()))#np.array(rox.R2rpy(rvec_err1))
            rvec_err1 = k*theta

            observed_tvec_difference = fixed_marker_transform.p - payload_marker_transform.p
            tvec_err1 = -fixed_marker_transform.R.dot(self.desired_transform.p) - observed_tvec_difference 
            # Map error to the robot spatial velocity
            world_to_camera_tf = self.tf_listener.lookupTransform("world", "gripper_camera_2", rospy.Time(0))
            camera_to_link6_tf = self.tf_listener.lookupTransform("gripper_camera_2","link_6", rospy.Time(0))
            
            t21 = -np.dot(rox.hat(np.dot(world_to_camera_tf.R,(camera_to_link6_tf.p-payload_marker_transform.p.reshape((1,3))).T)),world_to_camera_tf.R)#np.zeros((3,3))#
            
            # v = R_oc(vc)c + R_oc(omeega_c)_c x (r_pe)_o = R_oc(vc)c - (r_pe)_o x R_oc(omeega_c)_c
            tvec_err = t21.dot(rvec_err1).reshape((3,)) + world_to_camera_tf.R.dot(tvec_err1).reshape((3,))
            # omeega = R_oc(omeega_c)_c
            rvec_err = world_to_camera_tf.R.dot(rvec_err1).reshape((3,))
            
            tvec_err = np.clip(tvec_err, -0.2, 0.2)
            rvec_err = np.clip(rvec_err, -np.deg2rad(5), np.deg2rad(5))
                
            if tvec_err[2] <0.01:
                rospy.loginfo("Only Compliance Control.")
                tvec_err[2] = 0
            
            rot_err = rox.R2rpy(error_transform.R)
            rospy.loginfo("tvec difference: %f, %f, %f",error_transform.p[0],error_transform.p[1],error_transform.p[2])
            rospy.loginfo("rvec difference: %f, %f, %f",rot_err[0],rot_err[1],rot_err[2])
            
            
            dx = -np.concatenate((rvec_err, tvec_err))*self.K_pbvs


            print (error_transform.p[2]>0.01) , (np.linalg.norm([error_transform.p[0],error_transform.p[1]]) > self.stage3_tol_p), (np.linalg.norm(rox.R2rpy(error_transform.R)) > self.stage3_tol_r)
            print np.linalg.norm([error_transform.p[0],error_transform.p[1]]) , self.stage3_tol_p
            print np.linalg.norm(rox.R2rpy(error_transform.R)),self.stage3_tol_r

            # Compliance Force Control
            if(not self.ft_flag):
                raise Exception("haven't reached FT callback")
            # Compute the external force    
            FTread = self.FTdata - self.FTdata_0  # (F)-(F0)
            rospy.loginfo('================ FT1 =============:' + str(FTread))
            print '================ FT2 =============:', self.FTdata    
            
            if FTread[-1]> (self.F_d_set1+50):
                F_d = self.F_d_set1                   
            else:
                F_d = self.F_d_set2

            if (self.FTdata==0).all():
                rospy.loginfo("FT data overflow")
                dx[-1] += self.K_pbvs[5]*0.004
            else:
                tx_correct = 0
                if abs(self.FTdata[0])>90:
                    tx_correct = 0.0001*(abs(self.FTdata[0])-90)
                
                Vz = self.Kc*(F_d - FTread[-1]) + tx_correct
                dx[-1] = dx[-1]+Vz
          
            print "dx:", dx   
            
            current_joint_angles = self.controller_commander.get_current_joint_values()
            joints_vel = QP_abbirb6640(np.array(current_joint_angles).reshape(6, 1),np.array(dx))
            goal = self.trapezoid_gen(np.array(current_joint_angles) + joints_vel.dot(1),np.array(current_joint_angles),0.01,0.01,0.015)#acc,dcc,vmax)

            print "joints_vel:", joints_vel   

            self.client.wait_for_server()
            self.client.send_goal(goal)
            self.client.wait_for_result()
            res = self.client.get_result()
            if (res.error_code != 0):
                raise Exception("Trajectory execution returned error")

            FT_data_ori.append(self.FTdata )
            FT_data_biased.append(FTread)
            err_data_p.append(error_transform.p)
            err_data_rpy.append(rot_err)
            joint_data.append(current_joint_angles)
            time_data.append(time.time())


        filename_pose = "/home/rpi-cats/Desktop/YC/Data/Panel2_Placement_In_Nest_Pose_"+str(time.time())+".mat"
        scipy.io.savemat(filename_pose, mdict={'FT_data_ori':FT_data_ori, 'FT_data_biased':FT_data_biased, 
        'err_data_p':err_data_p, 'err_data_rpy': err_data_rpy, 'joint_data': joint_data, 'time_data': time_data})
        cv2.destroyWindow("")
        cv2.destroyWindow("transform")

        rospy.loginfo("End  ====================")
        
        ### End of initial pose 
        
def plot_trajectory_msg(trajectory_msg):    
    
    trajectory_command_t = np.array([p.time_from_start.to_sec() for p in trajectory_msg.points])
    trajectory_command_v = np.rad2deg(np.array([p.positions for p in trajectory_msg.points]))
    
    plt.close()
    
    plt.figure()
    for i in xrange(6):
        plt.subplot(2,3,i+1) 
        plt.plot(trajectory_command_t, trajectory_command_v[:,i], 'xr')
        #pchip = PchipInterpolator(trajectory_command_t,trajectory_command_v[:,i])
        #a=np.linspace(trajectory_command_t[0], trajectory_command_t[-1], 5000)
        #plt.plot(a, pchip(a))
        plt.title("Trajectory Joint %i"% i)
    
    
    plt.show(True)
    raw_input("Press enter to continue")

if __name__ == '__main__':
    main()
