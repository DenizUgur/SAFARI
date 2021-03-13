"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import math
from time import time as t

from rospy.core import deprecated

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Vector3, Twist
from std_msgs.msg import Header

from .cubic_spline_planner import *


k = 0.3  # control gain
Kp = 1.0  # speed proportional gain
L = 0.23  # [m] Wheel base of vehicle
max_steer = np.radians(20.0)  # [rad] max steering angle


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, pos, v=0.0):
        """Instantiate the object."""
        self.x, self.y, self.yaw = pos
        self.target_yaw = None
        self.v = v


def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


@deprecated
def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return (
        delta,
        current_target_idx,
    )


def basic_control(state, cx, cy, cyaw, last_target_idx, target_idx):
    """
    Basic steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # * current_target_idx for CTE
    # * target_idx for heading error

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


class Controller:
    def __init__(self, target_speed, cmd_topic) -> None:
        self.target_speed = target_speed
        self.state = None
        self.path = None
        self.last_idx = np.inf

        self.path_pub = rospy.Publisher("/safari/path", Path, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)

        self.tprev = t()
        self.dt = 0.1

    def forward(self):
        if self.state is None:
            return -1  # No state

        # Update state
        self.dt = t() - self.tprev
        self.tprev = t()
        self.state.v = (
            np.hypot(self.state.x - self.pos[0], self.state.y - self.pos[1]) / self.dt
        )
        self.state.x = self.pos[0]
        self.state.y = self.pos[1]
        self.state.yaw = self.pos[2]

        if self.last_idx >= self.target_idx:
            return -2  # Target reached

        cx, cy, cyaw = self.path
        ai = pid_control(self.target_speed, self.state.v)
        di, self.last_idx = basic_control(
            self.state, cx, cy, cyaw, self.last_idx, self.target_idx
        )
        self.state.target_yaw = di

        assert self.state.v < 100, "Robot is not responding"

        return 0  # Path tracking increment

    def apply(self):
        cx, cy, cr = self.pos
        tx, ty, tr = self.state.x, self.state.y, self.state.yaw

        delta_distance = math.hypot(ty - cy, tx - cx)

        ANG_vel = self.state.target_yaw
        LIN_vel = self.state.v

        rospy.logwarn_throttle(
            0.2,
            "Err_a={:.3f} :: Err_d={:.3f} :: V_a={:.3f} :: V_l={:.3f}".format(
                tr - cr, delta_distance, ANG_vel, LIN_vel
            ),
        )
        msg = Twist()
        linear = Vector3()
        angular = Vector3()

        linear.x = LIN_vel
        linear.y = 0
        linear.z = 0
        angular.x = 0
        angular.y = 0
        angular.z = ANG_vel

        msg.linear = linear
        msg.angular = angular
        # self.cmd_pub.publish(msg)

    def update_pos(self, pos):
        x, y, z, w = (
            pos.pose.pose.orientation.x,
            pos.pose.pose.orientation.y,
            pos.pose.pose.orientation.z,
            pos.pose.pose.orientation.w,
        )
        t1 = 2.0 * (w * z + x * y)
        t2 = 1.0 - 2.0 * (y ** 2 + z ** 2)
        self.pos = (
            pos.pose.pose.position.x,
            pos.pose.pose.position.y,
            math.atan2(t1, t2),
        )

        if self.state is None and self.path is not None:
            self.state = State(self.pos, v=0.0)
            cx, cy, _ = self.path
            self.target_idx, _ = calc_target_index(self.state, cx, cy)

    def update_path(self, path):
        cx, cy, cyaw, ck, s = calc_spline_course(path[:, 0], path[:, 1], ds=0.2)
        self.last_idx = -1
        self.path = (cx, cy, cyaw)
        self.publish_path()

    def publish_path(self):
        cx, cy, cyaw = self.path

        msg = Path()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        path = np.hstack(
            (
                np.array(cx).reshape(-1, 1),
                np.array(cy).reshape(-1, 1),
                np.array(cyaw).reshape(-1, 1),
            )
        )

        for x, y, r in path:
            pose = PoseStamped()
            pose.header = msg.header

            # set position
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 2.0

            # set orientation
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(r / 2)
            pose.pose.orientation.w = math.cos(r / 2)

            # add to path
            msg.poses.append(pose)

        self.path_pub.publish(msg)
