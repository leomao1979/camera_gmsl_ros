#ifndef LM_COMPRESSED_IMAGE_PUBLISHER 
#define LM_COMPRESSED_IMAGE_PUBLISHER

#include <ros/ros.h>
#include <string>

class LMCompressedImagePublisher {

public:
	LMCompressedImagePublisher(const std::string& topic);
	void publish(uint8_t *compressed_image, size_t size);

	ros::NodeHandle nh;
	ros::Publisher image_pub; 
};

#endif

