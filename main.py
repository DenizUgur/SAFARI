import rospy
import sys
import traceback
import argparse
import yaml
from scipy import spatial
from timeit import default_timer as dt
import numpy as np

# Elevation
from grid_map_msgs.srv import GetGridMap

# Import internals
from internal.mess import MESS
from internal.pf import *
from internal.driver.stanley_controller import Controller

# Import types
from nav_msgs.msg import Odometry


def main(args):
    # System Initialization
    rospy.init_node("safari")

    config = yaml.load(open("./config/config.yaml"), Loader=yaml.FullLoader)

    # Initialize Modules
    mess = MESS(np.load("config/parameters.npy"), 100)
    controller = Controller(5.0, config["cmd_topic"])
    get_submap = rospy.ServiceProxy("/elevation_mapping/get_submap", GetGridMap)

    # Subscribe to relevant topics and services
    rospy.Subscriber(config["odometry_topic"], Odometry, controller.update_pos)

    # Wait for elevation map to be available
    rospy.loginfo("Waiting for topics to become online...")
    rospy.wait_for_service("/elevation_mapping/get_submap", timeout=60)
    rospy.wait_for_message(config["odometry_topic"], Odometry, timeout=10)
    rospy.loginfo("OK!")

    try:
        # ROS Loop
        rospy.loginfo("Running the control loop")
        while not rospy.core.is_shutdown():

            # * Get submap from elevation_mapping
            start = dt()
            cx, cy, xsize, ysize = (
                controller.pos[0],
                controller.pos[1],
                config["local_size"],
                config["local_size"],
            )
            payload = get_submap(
                "odom",
                cx,
                cy,
                xsize,
                ysize,
                ["elevation"],
            ).map
            raw = payload.data[0]
            terrain = np.array(raw.data, dtype=float)
            terrain.shape = (raw.layout.dim[0].size, raw.layout.dim[1].size)
            terrain = np.rot90(terrain, k=2)
            rospy.logwarn(f"DEM Acqusition took {dt() - start:.4f} seconds")

            # * Process it
            start = dt()
            extent = (
                cx - xsize // 2,
                cy - ysize // 2,
                cx + xsize // 2,
                cy + ysize // 2,
            )
            mess.update_L2_ref(terrain, extent)
            mess.update_local(terrain, extent)
            mess.process(no_merge=True)
            rospy.logwarn(f"Mesh Process took {dt() - start:.4f} seconds")

            # * Create path
            start = dt()
            v, f, e, c = mess.map_L2
            mask = v[:, 2] < MESS.INVALID_EDGE * 0.3
            v_4tree = np.copy(v)[mask]

            # * First find point in masked area
            tree = spatial.KDTree(v_4tree[:, [0, 1]])
            _, si = tree.query((cx, cy))
            _, ei = tree.query((cx, cy + 20))  # TODO: Change to move goal

            # * Then find in all vertices
            sx, sy, _ = v_4tree[si]
            ex, ey, _ = v_4tree[ei]
            tree = spatial.KDTree(v[:, [0, 1]])
            _, si = tree.query((sx, sy))
            _, ei = tree.query((ex, ey))

            NM = NavMesh(v, e, c, mask)
            try:
                path, _ = path_finder(NM, si, ei)
            except Exception as err:
                print(err)
                continue
            controller.update_path(path)
            rospy.logwarn(f"Path Finding took {dt() - start:.4f} seconds")

            # * Follow path
            # TODO: Seperate into a deamon thread
            # print("Calculating")
            ret = controller.forward()
            print(ret)
            # if ret == 0:
            #     # print("Applying")
            #     # print(controller.pos)
            #     controller.apply()
            # elif ret == -2:
            #     print("reached")
            # TODO: Pause resource intensive tasks

    except (Exception, rospy.ROSException, KeyboardInterrupt):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        rospy.logfatal("Program crashed or halted")
        traceback.print_exception(
            exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
        )
        rospy.core.signal_shutdown("exited")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAFARI")
    parser.add_argument(
        "--dryrun", help="Do not start the control loop", action="store_true"
    )
    main(parser.parse_args())
