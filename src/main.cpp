// boost includes, useless if I am not ussing UDP
#include <boost/asio.hpp>
#include <boost/array.hpp>

// ros includes
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Pose2D.h>  

// opencv includes
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/calib3d.hpp"

// basic c++ includes
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <fstream>

// includes for the XIMEA API
#include <xiApiPlusOcv.hpp>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.0239f; //meters
const float arucoSquareDimension = 0.03525f; //to be changed
const Size chessboardDimensions = Size(6,9);

void createArucoMarkers()
{
	Mat outputMarker;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	for(int i = 0; i < 50; i++)
	{
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "4x4Marker_";
		convert << imageName << i << ".jpg";
		imwrite(convert.str(), outputMarker);
	}

}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for(int i = 0; i < boardSize.height; i++)
	{
		for(int j=0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for(vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}
		if(showResults)
		{
			drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
			imshow("looking for Corners",*iter);
			waitKey(0);
		}
	}
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);
	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8,1,CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if(outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns =cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;


		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r,c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				double value =  distanceCoefficients.at<double>(r,c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}
	return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ifstream inStream(name);
	if(inStream)
	{
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r,c) = read;
				cout << cameraMatrix.at<double>(r,c) << "\n";

			}
		}

		//distance coefficients
		inStream >> rows;
		inStream >> columns;
		
		distanceCoefficients = Mat::zeros(rows,columns,CV_64F);

		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				double read = 0.0f;
				distanceCoefficients.at<double>(r,c) = read;
				cout << distanceCoefficients.at<double>(r,c) << "\n";
			}
			inStream.close();
			return true;
		}
		return false;

	}
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension, ros::Publisher* myPub)
{
	//matrices for computing the rotation and translation between the frames
	
	geometry_msgs::Pose2D msg;

	Mat R1, R2, R, r;
	Mat t1, t2, t;
	
	Mat frame;

	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;

	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
	
	VideoCapture vid(1);

	if(!vid.isOpened())
	{
		return -1;
	}

	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	vector<Vec3d> rotationVectors, translationVectors;

	while(true)
	{
		
		if(!vid.read(frame))
			break;
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);

		for(int i = 0; i < markerIds.size(); i++)
		{
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f); //0.1f is the length of the axis
		}
		imshow("Webcam", frame);

		//Get the relative angle and translation of the 2 frames
		
		if(markerIds.size() >= 2)
		{
			Rodrigues(rotationVectors[0], R1);
			Rodrigues(rotationVectors[1], R2);
			R = R2.inv()*R1;
			Rodrigues(R,r);

			// z componant of r is actually the angle for the HoverCraft
			//cout << r.at<double>(2,0) << endl;
			Mat t1(translationVectors[0], false);
			Mat t2(translationVectors[1], false);
			cout << R1.inv()*(t2-t1) << endl;
			t = t2 - t1;

		//publish the ros message
		msg.x =(translationVectors[0]-translationVectors[1])[0];
		msg.y =(translationVectors[0]-translationVectors[1])[1];
		msg.theta = r.at<double>(2,0);

		myPub -> publish(msg);
		
		//cout << "Ids of marker : " << markerIds[0] <<"-----------------------" << markerIds[1] <<"---------------"<< markerIds[2] << endl;

		}

		
		if(waitKey(30) >= 0) break;
	}
	return 1;
			
}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
	//createArucoMarkers();
	Mat frame;
	Mat drawToFrame;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(1);

	if(!vid.isOpened())
	{
		return;
	}

	int framesPerSecond = 60;

	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	while(true)
	{
		if(!vid.read(frame))
			break;	

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
		if(found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);
		char character = waitKey(1000/framesPerSecond);

		switch(character)
		{
			case ' ':
				//save image
				if(found)
				{
					Mat temp;
					frame.copyTo(temp);
					savedImages.push_back(temp);
				}
				break;
			//case 27:
			case 10 :
				//ASCII for enter, start calibration, ASCII character for enter is 13 or 10?
				if(savedImages.size() > 15)
				{
					cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);		
					saveCameraCalibration("ILoveCameraCalibration", cameraMatrix, distanceCoefficients);

				}
				break;
			case 27:
				//ASCII for esc
				return;
				break;
		}
	}
	
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "vision");
	ros::NodeHandle n;
	ros::Publisher vision_pub = n.advertise<geometry_msgs::Pose2D>("vision",1000);

	Mat cameraMatrix = Mat::eye(3,3,CV_64F);
	Mat distanceCoefficients;

	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
	loadCameraCalibration("ILoveCameraCalibration", cameraMatrix, distanceCoefficients);
	startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimension, &vision_pub);
			

	return 0;
}
