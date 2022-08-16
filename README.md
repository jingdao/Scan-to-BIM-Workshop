Tutorial on Scan-to-BIM using Python and Open3D
---

In this tutorial, we will be using open-source software tools to create Building Information Models (BIM) from 3D point clouds. We will use simple functions in Python and Open3D to reconstruct building walls and floors from sensor data. This notebook will provide step-by-step instructions on how to implement common point clouds processing algorithms such as filtering, normal estimation, clustering, and robust parameter estimation.

*This tutorial is presented at a workshop in conjunction with the 9th International Conference on Construction Engineering and Project Management (ICCEPM 2022)*

**NumPy Library**

NumPy is an open-source Python library for numerical computing. Documentation for NumPy can be found [here](https://numpy.org/doc/stable/user/basics.html).

**Open3D Library**

Open3D is an open-source library that supports rapid development of software that deals with 3D data. Documentation for using Open3D can be found [here](http://www.open3d.org/docs/release/getting_started.html).

**CloudCompare viewer**

[CloudCompare](https://www.danielgm.net/cc/) is an open-source 3D point cloud processing and visualization software. CloudCompare supports many open point cloud formats (ASCII, LAS, E57, etc.) as well as triangular meshes (OBJ, PLY, STL, FBX, etc.). The intermediate outputs from this notebook can be viewed and processed in CloudCompare.
