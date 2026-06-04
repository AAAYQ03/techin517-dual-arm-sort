"""Fetch joint position limits from a URDF published on a robot_description topic.

Subscribes once with ``durability=TRANSIENT_LOCAL`` (matching
``robot_state_publisher``'s default), parses the URDF with
``urdf_parser_py``, and returns a ``{joint_name: (lower, upper)}`` dict.
"""

import time
from typing import Dict, Iterable, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String
from urdf_parser_py.urdf import URDF


def fetch_joint_limits(
    node: Node,
    joint_names: Iterable[str],
    topic: str = '/follower/robot_description',
    timeout_s: float = 5.0,
) -> Dict[str, Tuple[float, float]]:
    """Block until ``topic`` produces a URDF, then return joint limits.

    Pumps the node's executor with ``rclpy.spin_once`` so this can be called
    from ``__init__`` before the main executor starts spinning.

    Raises:
        TimeoutError: if no message arrives within ``timeout_s``.
        KeyError: if any requested joint is missing from the URDF.
        ValueError: if a joint has no ``<limit>`` element.
    """
    qos = QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
    )

    received = []

    def cb(msg: String):
        if not received:
            received.append(msg.data)

    sub = node.create_subscription(String, topic, cb, qos)
    try:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not received:
            raise TimeoutError(
                f'no message on {topic} within {timeout_s:.1f}s '
                '(is robot_state_publisher running?)'
            )
    finally:
        node.destroy_subscription(sub)

    urdf = URDF.from_xml_string(received[0])
    limits: Dict[str, Tuple[float, float]] = {}
    for name in joint_names:
        joint = urdf.joint_map.get(name)
        if joint is None:
            raise KeyError(f'joint {name!r} not present in URDF')
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f'joint {name!r} has no <limit lower=... upper=...>')
        limits[name] = (float(joint.limit.lower), float(joint.limit.upper))
    return limits
