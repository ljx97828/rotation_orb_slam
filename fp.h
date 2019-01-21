#ifndef _FP_H
#define _FP_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2\video\video.hpp>
using namespace Eigen;

using namespace std;
using namespace cv;


Point2d pixel2cam(const Point2d& p, const Mat& K);
void find_feature_matches(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, vector<DMatch> &matches);
void pose_estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector< DMatch > matches, Mat& R, Mat& t);
void triangulation(const vector<KeyPoint>& keypoint_1, const vector<KeyPoint>& keypoint_2, const std::vector< DMatch >& matches, const Mat& R, const Mat& t, vector<Point3d>& points);
#endif // !_FP_H

