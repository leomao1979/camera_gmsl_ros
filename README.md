# camera_gmsl_raw_ros

This sample captures RAW images from GMSL camera, convert to processed then
publish to ROS.

1) Run "./camera_gmsl_raw_ros" without any option:
   Publish uncompressed image to the default ROS topic "/camera/image"

2) Run "./camera_gmsl_raw_ros --ros-topic='ros_topic_name'"
   If 'ros_topic_name' starts with '/camera/image/compressed', publish JPEG compressed image to 'ros_topic_name'. 
   Otherwise, publish original uncompressed image.
   
