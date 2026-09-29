"""
SPARK Real - Standalone Python pipeline for real-world robot control.

No ROS2 dependency. The deployed default is the Franka FR3 (franky/libfranka);
UR10e (ur_rtde) and Unitree G1 families are also supported. Uses:
- SAM3 for perception
- Gemini/OpenAI for task planning
- VLA policy control (optional)
"""
