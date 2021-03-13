import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def callback(msg):
    new_msg = PoseWithCovarianceStamped()
    new_msg.header.stamp = msg.header.stamp
    new_msg.header.frame_id = msg.child_frame_id
    new_msg.pose = msg.pose

    pub.publish(new_msg)


rospy.init_node("pose_relay")

sub = rospy.Subscriber("/odometry/filtered", Odometry, callback, tcp_nodelay=True)
pub = rospy.Publisher("/odometry/pose", PoseWithCovarianceStamped, queue_size=1)

rospy.spin()
