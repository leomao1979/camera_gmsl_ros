#include "LMCompressedImagePublisher.hpp"
#include <sensor_msgs/CompressedImage.h>

using namespace std;

LMCompressedImagePublisher::LMCompressedImagePublisher(const string& topic) {
	image_pub = nh.advertise<sensor_msgs::CompressedImage>(topic, 1);
}

void LMCompressedImagePublisher::publish(uint8_t* compressed_image, size_t size) {
	sensor_msgs::CompressedImage c_img_msg;
	std_msgs::Header header;
	header.stamp = ros::Time::now();
	c_img_msg.header = header;
	c_img_msg.format = "png";
	c_img_msg.data.resize(size);
	memcpy(c_img_msg.data.data(), compressed_image, size);
	// publish to ros
	image_pub.publish(c_img_msg);
}
