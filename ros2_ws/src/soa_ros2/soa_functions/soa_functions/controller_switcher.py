"""Controller switcher service node.

Provides the ~/switch_controller service (soa_interfaces/srv/SwitchController)
to hot-swap between JointTrajectoryController (MoveIt) and
ForwardCommandController (Rosetta) without restarting hardware.

Both controller sets must already be loaded (spawned active/inactive) by the
bringup launch. This node only activates/deactivates them via the
controller_manager/switch_controller service.

Launch in the /follower namespace (soa_bringup.launch.py handles this).

Services:
    ~/switch_controller (soa_interfaces/srv/SwitchController)
        request:  controller_type — 'jtc' or 'forward'
        response: success, current_controller

Topics (published):
    ~/current_controller (std_msgs/String, TRANSIENT_LOCAL)
        Always holds the current mode; late subscribers receive the last value.

Parameters:
    initial_mode (str, default 'forward')
        Must match the controller set activated at bringup time.
"""

import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from builtin_interfaces.msg import Duration
# TODO: import SwitchController from controller_manager_msgs.srv
# aliasing it as CmSwitchController to avoid colliding with the
# soa_interfaces.srv.SwitchController already imported below
from soa_interfaces.srv import SwitchController
from std_msgs.msg import String


class ControllerSwitcherNode(Node):
    JTC = ['arm_controller', 'gripper_controller']
    FCC = ['arm_fwd_controller', 'gripper_fwd_controller']
    VALID = {'jtc': JTC, 'forward': FCC}

    def __init__(self):
        super().__init__('controller_switcher')
        self.declare_parameter('initial_mode', 'forward')
        self._mode = self.get_parameter('initial_mode').value

        # ReentrantCallbackGroup allows the service callback to await a client
        # future while the executor's second thread processes the response
        self._cb_group = ReentrantCallbackGroup()

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._mode_pub = self.create_publisher(String, '~/current_controller', qos)

        self._cm_client = self.create_client(
            # TODO: create a client for the controller manager controller switcher service
            # make sure to use the correct namespace for the controller manager service
        )

        self.create_service(
            # TODO: create the ~/switch_controller service using SwitchController (soa_interfaces)
            # wiring it to _handle_switch with the reentrant callback group
            # note: ~ expands to the node's full name
            # e.g. /follower/controller_switcher/switch_controller
            # you can verify with ros2 service list after launching the node
        )
        self._publish_mode()
        self.get_logger().info(f'Controller switcher ready. Current mode: {self._mode}')

    def _handle_switch(self, req, res):
        if req.controller_type not in self.VALID:
            self.get_logger().warn(
                f"Unknown controller_type '{req.controller_type}'. Use 'jtc' or 'forward'."
            )
            res.success = False
            res.current_controller = self._mode
            return res

        if req.controller_type == self._mode:
            res.success = True
            res.current_controller = self._mode
            return res

        activate = self.VALID[req.controller_type]
        deactivate = self.VALID['forward' if req.controller_type == 'jtc' else 'jtc']

        ok = self._call_switch(activate, deactivate)
        if ok:
            self._mode = req.controller_type
            self._publish_mode()
            self.get_logger().info(f'Switched to {self._mode} mode.')
        else:
            self.get_logger().error('controller_manager/switch_controller call failed.')

        res.success = ok
        res.current_controller = self._mode
        return res

    def _call_switch(self, activate: list, deactivate: list) -> bool:
        if not self._cm_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('controller_manager/switch_controller service not available.')
            return False

        req = CmSwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = CmSwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=5, nanosec=0)

        event = threading.Event()
        result: list = [None]

        def _done(future):
            result[0] = future
            event.set()

        future = # TODO: use the controller manager client to switch controllers
                 # use call_async to not block this thread
        future.add_done_callback(_done)

        if not event.wait(timeout=10.0):
            self.get_logger().error('switch_controller call timed out.')
            return False
        if result[0].result() is None:
            self.get_logger().error('switch_controller returned no result.')
            return False
        return result[0].result().ok

    def _publish_mode(self):
        msg = String()
        msg.data = self._mode
        self._mode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerSwitcherNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
