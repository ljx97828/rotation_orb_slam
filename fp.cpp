#include "fp.h"

void find_feature_matches(Mat img_1, Mat img_2, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, vector<DMatch> &matches)
{
	Mat K = (Mat_<double>(3, 3) << 530.046777935726, 0, 361.513776785628, 0, 532.158546628743, 222.389370638336, 0, 0, 1);

	Mat descriptors_1, descriptors_2;
	Ptr<ORB> orb = ORB::create(100, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);

		orb->compute(img_1, keypoints_1, descriptors_1);
		orb->compute(img_2, keypoints_2, descriptors_2);

		Mat outimg1;
		drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//imshow("ORB特征点", outimg1);

		
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(descriptors_1, descriptors_2, matches);

		double min_dist = 10000, max_dist = 0;
		for (int i = 0; i < descriptors_1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		vector< DMatch > good_matches;
		for (int i = 0; i < descriptors_1.rows; i++)
		{
			if (matches[i].distance <= max(2 * min_dist, 50.0))
			{
				good_matches.push_back(matches[i]);
			}
		}
		Mat img_match;
		Mat img_goodmatch;
		drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
		matches = good_matches;

		drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
		//imshow("所有匹配点对", img_match);
		imshow("优化后匹配点对", img_goodmatch);
		//cout << "一共找到了" << matches.size() << "组匹配点" << endl;
	
	

}

Point2d pixel2cam(const Point2d& p, const Mat& K)
{
	return Point2d
	(
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	);

}

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector< DMatch > matches, Mat& R, Mat& t)
{
	Mat K = (Mat_<double>(3, 3) << 530.046777935726, 0, 361.513776785628, 0, 532.158546628743, 222.389370638336, 0, 0, 1);

	vector<Point2f> points1;
	vector<Point2f> points2;

	for (int i = 0; i < (int)matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}
	
	Mat fundamental_matrix;
	fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
	//cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

	Point2d principal_point(361.513776785628, 222.389370638336);
	int focal_length = 530.046777935726;

	Mat essential_matrix;
	essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);
	//cout << "essential_matrix is " << endl << essential_matrix << endl;

	Mat homography_matrix;
	homography_matrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
	//cout << "homography_matrix is " << endl << homography_matrix << endl;
	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	//cout << "R is " << endl << R << endl;
	//cout << "t is " << endl << t << endl;
}

void triangulation(const vector<KeyPoint>& keypoint_1, const vector<KeyPoint>& keypoint_2, const std::vector< DMatch >& matches, const Mat& R, const Mat& t, vector<Point3d>& points)
{
	Mat K = (Mat_<double>(3, 3) << 530.046777935726, 0, 361.513776785628, 0, 532.158546628743, 222.389370638336, 0, 0, 1);

	Mat T1 = (Mat_<double>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	Mat T2 = (Mat_<double>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);
	vector<Point2d> pts_1, pts_2;
	for (DMatch m : matches)
	{
		pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
		pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
	}

	Mat pts_4d;
	triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
	Mat x;
	for (int i = 0; i < pts_4d.cols; i++)
	{
		//x = pts_4d.col(i);
		//cout << x;
		//x /= x.at<float>(0, 3);
		double gu = pts_4d.col(i).at<double>(3, 0);
		Point3d p(
			pts_4d.col(i).at<double>(0, 0)/ pts_4d.col(i).at<double>(3, 0),
			pts_4d.col(i).at<double>(1, 0) / pts_4d.col(i).at<double>(3, 0),
			pts_4d.col(i).at<double>(2, 0) / pts_4d.col(i).at<double>(3, 0)
		);
		points.push_back(p);
	}

}