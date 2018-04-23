#include "LMCompressedImagePublisher.hpp"

#include <sensor_msgs/CompressedImage.h>

LMCompressedImagePublisher::LMCompressedImagePublisher(const std::string& topic) {
	image_pub = nh.advertise<sensor_msgs::CompressedImage>(topic, 1);
}

void LMCompressedImagePublisher::publish(uint8_t* compressed_image, const std::string& format, size_t size) {
	sensor_msgs::CompressedImage c_img_msg;
	std_msgs::Header header;
	header.stamp = ros::Time::now();
	c_img_msg.header = header;
	c_img_msg.format = format;
	c_img_msg.data.resize(size);
	memcpy(c_img_msg.data.data(), compressed_image, size);
	// publish to ros
	image_pub.publish(c_img_msg);
}
