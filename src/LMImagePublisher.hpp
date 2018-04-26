#ifndef LM_IMAGE_PUBLISHER 
#define LM_IMAGE_PUBLISHER

#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>

class LMImagePublisher {

public:
    LMImagePublisher(const std::string& topic);
    // data: RGB format
    void publish(uint8_t *data, const ros::Time& stamp, int width, int height);

    ros::NodeHandle m_nh;
    image_transport::ImageTransport m_it;
    image_transport::Publisher m_imagePub;
    int m_seq;
};

#endif
