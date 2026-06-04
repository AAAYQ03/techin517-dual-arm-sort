"""Save joint states service node.

Provides the /follower/save_joint_states service (soa_interfaces/srv/SaveJointStates)
to capture the current joint states of the follower arm and optionally append them
to a CSV file for later analysis or replay.

The joint state positions are recorded with column headers derived from the joint
names in the JointState message.

Can be run standalone without a namespace argument:
    ros2 run soa_functions save_joint_states

Services:
    /follower/save_joint_states (soa_interfaces/srv/SaveJointStates)
        request:  csv_path — path to CSV file; if empty, joint states are not saved
        response: success, joint_states

Subscriptions:
    /follower/joint_states (sensor_msgs/JointState)
"""

# TODO: read through the official ros humble documentation to find the definition of the joint states message type
# TODO: find and read through the SaveJointStates service in soa_interfaces

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# TODO: import the standard JointState message type
# TODO: import the SaveJointStates service from soa_interfaces


class SaveJointStatesNode(Node):

    def __init__(self):
        super().__init__('save_joint_states')

        self._latest_js: JointState | None = None
        self._cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            # TODO: subscribte to the follower's joint_states topic
            #       make sure to use the correct namespace
            #       write a function to save the latest joint states from message to self._latest_js
            #       use the callback group defined above
        )

        self.create_service(
            # TODO: define the save_joint_states service
            #       make sure to use the correct namespace
            #       use the save_joint_states function defined below
            #       use the callback group defined above
        )

        self.get_logger().info('SaveJointStates service ready.')

    def _handle_save_joint_states(self, req, res):
        """Handle the /follower/save_joint_states service request.

        Captures the most recently received joint states and optionally writes
        them to a CSV file. Returns immediately with success=False if no joint
        state has been received yet.

        Args:
            req (SaveJointStates.Request): Service request containing:
                csv_path (str): Filesystem path to the target CSV file.
                    If empty, joint states are returned but not written to disk.
            res (SaveJointStates.Response): Service response to populate.

        Returns:
            SaveJointStates.Response: Populated response with:
                joint_states (sensor_msgs/JointState): The latest captured joint state.
                success (bool): True if joint states were captured (and written,
                    if csv_path was provided); False otherwise.
        """
        if self._latest_js is None:
            # TODO: handle the case where no joint states have been saved yet
            #       use the ros logger to print an informative message
            #       set the result (res) to false
            #       return the result

        # TODO: set res.joint_states equal to the latest joint states
        #       set res.success to true

        if req.csv_path:
            try:
                # TODO: use the function below to save the joint states to a csv
            except OSError as e:
                # TODO: if there is an error, use the ros logger to log an error
                #       set res.success to false

        return res

    def _append_to_csv(self, path: str, js: JointState) -> None:
        """Append a single row of joint positions to a CSV file.

        If the file does not yet exist, a header row is written first using the
        joint names from the JointState message. Each subsequent call appends
        one data row mapping joint names to their current positions.

        Args:
            path (str): Filesystem path to the target CSV file. The file is
                created if it does not exist; existing content is preserved.
            js (sensor_msgs/JointState): Joint state message whose ``name`` and
                ``position`` fields supply the column headers and row values.

        Returns:
            None

        Raises:
            OSError: If the file cannot be opened or written to.
        """
        # TODO: write the joint states to a csv file
        #       import any additional libraries you need
        #       check if the file exists
        #       open the file in append mode
        #       if the file did not exist:
        #           write a header line of the joint names
        #       write a row of joint values corresponding to the header joint names
        #       log the successful write using the ros logger


def main(args=None):
    rclpy.init(args=args)
    node = SaveJointStatesNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
