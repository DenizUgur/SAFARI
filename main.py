import yaml
import rospy
import signal
import argparse
import numpy as np
import stackprinter
from loguru import logger
import matplotlib.pyplot as plt
from timeit import default_timer as dt

# Elevation
from grid_map_msgs.srv import GetGridMap

# Import internals
from internal.pf import *
from internal.mess import MESS
from internal.driver.stanley_controller import Controller

# Import types
from nav_msgs.msg import Odometry

prev_cancel = 0


def signal_handler(sig, frame):
    global prev_cancel
    if abs(prev_cancel - dt()) < 1:
        logger.critical("User halted the program!")
        exit(1)
    else:
        logger.warning("Press ^C again to halt the program")
    prev_cancel = dt()


def main(args):
    # System Initialization
    rospy.init_node("safari")
    signal.signal(signal.SIGINT, signal_handler)

    config = yaml.load(open("./config/config.yaml"), Loader=yaml.FullLoader)

    # Initialize Modules
    mess = MESS(np.load("config/parameters.npy"), 96)
    controller = Controller(5.0, config["cmd_topic"])
    get_submap = rospy.ServiceProxy("/elevation_mapping/get_submap", GetGridMap)

    # Subscribe to relevant topics and services
    rospy.Subscriber(config["odometry_topic"], Odometry, controller.update_pos)

    # Wait for elevation map to be available
    logger.info("Waiting for topics to become online...")
    rospy.wait_for_service("/elevation_mapping/get_submap", timeout=60)
    rospy.wait_for_message(config["odometry_topic"], Odometry, timeout=10)
    logger.success("OK!")

    try:
        # ROS Loop
        logger.info("Running the control loop")
        plt_shown = False
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
            terrain = np.flip(terrain, axis=0)
            logger.warning(f"DEM Acqusition took {dt() - start:.4f} seconds")

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
            logger.warning(f"Mesh Process took {dt() - start:.4f} seconds")

            if plt_shown:
                plt.close()

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(6, 12), sharex=True, sharey=True
            )

            terrain[terrain == 0] = np.nan
            ax1.contourf(terrain, cmap="terrain", levels=200)
            ax2.contourf(mess.map_L1_ref, cmap="terrain", levels=200)

            fig.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt_shown = True

    except (Exception, rospy.ROSException, KeyboardInterrupt):
        logger.critical("Program crashed or halted")
        logger.info(":: TRACEBACK ::")
        stackprinter.show()
        rospy.core.signal_shutdown("exited")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAFARI")
    parser.add_argument(
        "--dryrun", help="Do not start the control loop", action="store_true"
    )
    main(parser.parse_args())
