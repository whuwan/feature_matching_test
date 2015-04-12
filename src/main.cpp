#include <iostream>
#include <ros/package.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main( int argc, char** argv )
{
  std::string data_dir = ros::package::getPath("feature_matching_test");
  cv::Mat img_1 = cv::imread(data_dir+"/data/img_1.png", 0 );
  cv::Mat img_2 = cv::imread(data_dir+"/data/img_2.png", 0 );

  if(!img_1.data || !img_2.data) {
    std::cerr << "ERROR: images not found in folder: " << data_dir << std::endl;
    return -1;
  }

  // detect keypoints.
  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("BRISK");
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // compute descriptors.
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("BRISK");
  cv::Mat descriptors_1, descriptors_2;
  extractor->compute(img_1, keypoints_1, descriptors_1 );
  extractor->compute(img_2, keypoints_2, descriptors_2 );

  // Matching descriptor vectors with a brute force matcher.
  cv::BFMatcher matcher(cv::NORM_HAMMING); //cv::NORM_HAMMING, cv::NORM_L2
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors_1, descriptors_2, matches);

  // Draw matches.
  cv::Mat img_matches;
  cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

  // Show detected matches.
  cv::imshow("Matches", img_matches );
  cv::waitKey(0);

  return 0;
}
