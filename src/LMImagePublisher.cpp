#include "LMImagePublisher.hpp"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#ifdef USE_CV_BRIDGE
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#endif

LMImagePublisher::LMImagePublisher(const std::string& topic) : it(nh) {
   	image_pub = it.advertise(topic, 1);
}

// data: RGB
void LMImagePublisher::publish(uint8_t* data, int width, int height) {
#ifdef USE_CV_BRIDGE
    cv::Mat rgb_img_mat(cv::Size(width, height), CV_8UC3, data);
    cv_bridge::CvImage cvImage = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_img_mat);
    image_pub.publish(cvImage.toImageMsg());
#else
	sensor_msgs::Image image_msg;
	std_msgs::Header header;
	header.stamp = ros::Time::now();
	image_msg.header = header;
	image_msg.encoding = sensor_msgs::image_encodings::RGB8;
	image_msg.height = height;
	image_msg.width = width;
	image_msg.step = width * 3;
	int size = width * height * 3;
	image_msg.data.resize(size);
	memcpy(image_msg.data.data(), data, size);
	image_pub.publish(image_msg);
#endif
}

