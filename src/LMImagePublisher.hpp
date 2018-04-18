#ifndef LM_IMAGE_PUBLISHER 
#define LM_IMAGE_PUBLISHER

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <string>

class LMImagePublisher {

public:
   LMImagePublisher(const std::string& topic);
   void publish(uint8_t *data, int width, int height);

   ros::NodeHandle nh;
   image_transport::ImageTransport it;
   image_transport::Publisher image_pub;
};

#endif

