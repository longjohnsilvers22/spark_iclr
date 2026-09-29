System Architecture: Bimanual Franka VLA Teleoperation Rig 

1. Primary Manipulators (Arms) 

Hardware: 1x Franka Emika Panda (Original, Left), 1x Franka Research 3 (FR3, Right). 

Control Interface: Both operate via the Franka Control Interface (FCI) using libfranka. 

Kinematics: Minor differences exist in the internal dynamic models and mass matrices between the Panda and FR3, which must be accounted for if using torque-level or sensitive impedance control. Both are treated as a unified 14-DoF system for motion planning to ensure a shared collision environment. 

2. End Effectors (Grippers) 

Hardware: Custom 3D-printed parallel jaw grippers (ALOHA project design). 

Actuation: Driven by internal Dynamixel servos. 

Communication: Interfaced to the host PC via a Source Robotics USB-to-CAN adapter (green PCB). 

3. Perception System (Cameras) 

Global/Third-Person View: ZED Mini (ZED M) stereoscopic camera mounted on a center tripod. Provides overarching workspace state, depth maps, and point clouds (ideal for 3D Gaussian Splatting or general geometry capture). 

Local/Egocentric View: Wrist-mounted cameras (standard ALOHA spec, likely Intel RealSense) attached to both arms. Provides high-resolution, occlusion-free visual features for fine manipulation and grasping tasks. 

4. Networking and State Coordination 

Physical Connection: Arms connect to the host PC via dedicated multi-port PCIe Network Interface Cards (NICs) to maintain the strict 1kHz real-time control loop. Direct connections are used to avoid switch-induced jitter. 

State Broadcasting: A low-level hardware loop (running in C++ or Python within an Ubuntu/ROS2 environment) reads joint states at 1kHz. 

Middleware: ZeroMQ (ZMQ) is utilized as a low-latency bridge to broadcast the concatenated state vector (arm joints + gripper states) to the neural network/policy inference nodes, bypassing heavier ROS middleware overhead for the ML components. 

5. Power and Safety Infrastructure 

Hardware: CyberPower 1500VA Uninterruptible Power Supply (UPS). 

Function: Protects the sensitive Franka controllers and joint brakes from voltage spikes and sudden power loss, allowing for safe, graceful shutdown of the control loops and hardware during an outage. 

 