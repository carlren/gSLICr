#include <time.h>
#include <stdio.h>
#include "FastImgSeg.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#ifdef DEBUG
#pragma comment( lib, "opencv_core2410d.lib" )
#pragma comment( lib, "opencv_highgui2410d.lib" )
#pragma comment( lib, "opencv_imgproc2410d.lib" )
// DEBUG
#else
#pragma comment( lib, "opencv_core2410.lib" )
#pragma comment( lib, "opencv_highgui2410.lib" )
#pragma comment( lib, "opencv_imgproc2410.lib" )
#endif

using namespace std;
using namespace cv;

void main()
{
	//VideoCapture cap("e:/data/video/bottle.avi");
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cerr << "no video cap!\n";
		return;
	}
	
	namedWindow("frame", WINDOW_AUTOSIZE);

	Mat oldFrame;
	Mat frame;
	Size s(640, 480);

	cap.read(oldFrame);
	resize(oldFrame, frame, s);

	
	FastImgSeg* mySeg=new FastImgSeg();
	mySeg->initializeFastSeg(frame.cols, frame.rows, 1200);

	unsigned char* imgBuffer = (unsigned char*)malloc(frame.cols*frame.rows*sizeof(unsigned char) * 4);
	memset(imgBuffer, 0, frame.cols*frame.rows * sizeof(unsigned char) * 4);

	while (cap.read(oldFrame))
	{
		resize(oldFrame, frame, s);
		
		// gSLIC currently only support 4-dimensional image 
		for (int i = 0; i<s.height; i++)
		{
			for (int j=0;j<s.width;j++)
			{
				int bufIdx=(i*s.width+j)*4;

				imgBuffer[bufIdx] = frame.at<cv::Vec3b>(i, j)[0];
				imgBuffer[bufIdx + 1] = frame.at<cv::Vec3b>(i, j)[1];
				imgBuffer[bufIdx + 2] = frame.at<cv::Vec3b>(i, j)[2];
			}
		}

		mySeg->LoadImg(imgBuffer);
		mySeg->DoSegmentation(XYZ_SLIC,0.3);
		mySeg->Tool_GetMarkedImg();

		for (int i = 0; i<s.height; i++)
		{
			for (int j = 0; j<s.width; j++)
			{
				int bufIdx = (i*s.width + j) * 4;

				frame.at<cv::Vec3b>(i, j)[0] = mySeg->markedImg[bufIdx];
				frame.at<cv::Vec3b>(i, j)[1] = mySeg->markedImg[bufIdx + 1];
				frame.at<cv::Vec3b>(i, j)[2] = mySeg->markedImg[bufIdx + 2];
			}
		}
		

		imshow("frame", frame);
		if( cvWaitKey(10) == 27 )
			break;
	}

	free(imgBuffer);
	cvDestroyWindow( "frame" );

}