# SAFARI

This paper presents a fast, energy-efficient, and low computational cost traversal solution on sloped terrain. The use of grid-based search algorithms requires high computational power and takes a long time because almost every point on the map is visited. An approach that does not depend on the global map but can also navigate towards the target can be presented as a new solution. A cost map for motion planning using depth field and color image data is formed in real-time. The proposed motion planning algorithm, named SAFARI, utilizes four cost layers to efficiently evaluate its surroundings. To reduce the computational overhead, only select features are evaluated and the rover's motion planning cycle speed is increased. SAFARI has been tested against path planning alternatives and has also been proven to work with simulations and field tests. This concept is expected to be used in space applications and cave exploration tasks.

# Purpose of this repository

In this repository we demonstrate how a incramental search on a terrain can speed up the path calculation. Furthermore, we simplify the grid-based map to a graph-based map while preserving the original map features.

# Citation

```
@INPROCEEDINGS{9551617,
  author={Ugur, Deniz and Bebek, Ozkan},
  booktitle={2021 IEEE 17th International Conference on Automation Science and Engineering (CASE)},
  title={Fast and Efficient Terrain-Aware Motion Planning for Exploration Rovers},
  year={2021},
  volume={},
  number={},
  pages={1561-1567},
  doi={10.1109/CASE49439.2021.9551617}}
```
