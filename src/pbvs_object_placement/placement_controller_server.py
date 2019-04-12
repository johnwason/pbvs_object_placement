from __future__ import absolute_import
#TODO: import placement controller class like shown in next line
from .pbvs_object_placement import PBVSPlacementController
from pbvs_object_placement.msg import PBVSPlacementAction, PBVSPlacementResult
import actionlib
import rospy
from safe_kinematic_controller.ros.commander import ControllerCommander
import traceback

class PlacementControllerServer(object):
    def __init__(self,controller_commander,urdf_xml_string,srdf_xml_string,camera_image,camera_trigger,camera_info):
        #TODO: Replace next line with creation of placement controller class, it should still be called controller 
        self.controller=PBVSPlacementController(controller_commander,urdf_xml_string,srdf_xml_string,camera_image,camera_trigger,camera_info)
        self.server=actionlib.ActionServer("placement_step", PBVSPlacementAction, goal_cb=self.execute_cb,cancel_cb=self.cancel, auto_start=False)
        
        self.server.start()
        self.previous_goal=None
        
    def cancel(self,goal):
        self.controller.abort()
        self.previous_goal.set_canceled()
        
    def execute_cb(self, goal):
        self.controller.goal_handle=goal
        goal.set_accepted()
        
        
        
        self.previous_goal=goal
        try:
            self.controller.set_goal(goal.get_goal())
            self.controller.pbvs_stage1()
            self.controller.pbvs_stage2()
            self.controller.pbvs_jacobian()
        except Exception as err:
            res = PBVSPlacementResult()
            res.error_msg=str(err)
            goal.set_aborted(res)
            traceback.print_exc()
            return
        try:
            self.controller.pbvs_stage3()
        except Exception as err:
            res = PBVSPlacementResult()
            self.controller.controller_commander.set_controller_mode(self.controller.controller_commander.MODE_HALT,0.7,[], [])
            res.error_msg=str(err)
            goal.set_succeeded(res)

            traceback.print_exc()
            return
        rospy.loginfo(goal.get_goal_status())
        
        res = PBVSPlacementResult()
        res.error_msg="succeeded"
        #res.state=self.controller.state
        #res.target=self.controller.current_target if self.controller.current_target is not None else ""
        #res.payload=self.controller.current_payload if self.controller.current_payload is not None else ""
        
        goal.set_succeeded(res)
        
            
def placement_controller_server_main():
    rospy.init_node("placement_controller_server")
    urdf_xml_string=rospy.get_param("robot_description")
    srdf_xml_string=rospy.get_param("robot_description_semantic")
    controller_commander=ControllerCommander()
    camera_image=rospy.get_param("~image_topic")
    camera_trigger=rospy.get_param("~camera_trigger")
    camera_info=rospy.get_param("~camera_info")
    
    s=PlacementControllerServer(controller_commander,urdf_xml_string,srdf_xml_string,camera_image,camera_trigger,camera_info)
    
    rospy.spin()
