#ifndef ROS_IMAGE_PUBLISHER_H
#define ROS_IMAGE_PUBLISHER_H

#include <string>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <image_transport/image_transport.h>

class ROSImagePublisher {

public:
    ROSImagePublisher(ros::NodeHandle &node, const std::string& topic, bool compress);
    // data: RGB format
    void publish_image(uint8_t *data, const ros::Time& stamp, int width, int height);
    // data: compressed image
    void publish_compressed_image(uint8_t *compressed_image, const ros::Time& stamp, const std::string& format, size_t size);

private:
    // Publisher for non-compressed image
    image_transport::ImageTransport m_it;
    image_transport::Publisher m_imagePub;

    // Publisher for compressed image
    ros::Publisher m_pub;
    int m_seq;
};

#endif
