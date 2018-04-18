#include "LMImagePublisher.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;

LMImagePublisher::LMImagePublisher(const string& topic) : it(nh) {
   	image_pub = it.advertise(topic, 1);
}

void LMImagePublisher::publish(uint8_t* data, int width, int height) {
	// Convert RGBA to RGB 
    cv::Mat rgba_img_mat(cv::Size(width, height), CV_8UC4, data);
    cv::Mat rgb_img_mat;
    cv::cvtColor(rgba_img_mat, rgb_img_mat, cv::COLOR_RGBA2RGB);
    cv_bridge::CvImage cvImage = cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_img_mat);
    image_pub.publish(cvImage.toImageMsg());
}

