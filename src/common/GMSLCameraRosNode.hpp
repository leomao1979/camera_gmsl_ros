#ifndef GMSL_CAMERA_ROS_NODE_H
#define GMSL_CAMERA_ROS_NODE_H

#include <string>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <image_transport/image_transport.h>

#include "framework/DriveWorksSample.hpp"
using namespace dw_samples::common;

class GMSLCameraRosNode {

public:
    GMSLCameraRosNode(DriveWorksSample* app, const std::string& topic, bool compress);
    // data: RGB format
    void publish_image(uint8_t *data, const ros::Time& stamp, int width, int height);
    // data: compressed image
    void publish_compressed_image(uint8_t *compressed_image, const ros::Time& stamp, const std::string& format, size_t size);

private:
    void processEnabled(const std_msgs::Bool::ConstPtr& msg);

    ros::NodeHandle m_nh;

    // Publisher for non-compressed image
    image_transport::ImageTransport m_it;
    image_transport::Publisher m_imagePub;

    // Publisher for compressed image
    ros::Publisher m_pub;
    int m_seq;

    DriveWorksSample* m_app;
};

#endif
