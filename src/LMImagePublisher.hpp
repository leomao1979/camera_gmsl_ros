#ifndef LM_IMAGE_PUBLISHER 
#define LM_IMAGE_PUBLISHER

#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>

class LMImagePublisher {

public:
    LMImagePublisher(const std::string& topic);
    // data: RGB format
    void publish(uint8_t *data, int width, int height);

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Publisher image_pub;
};

#endif
