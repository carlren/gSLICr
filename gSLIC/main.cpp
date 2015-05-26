#include <time.h>
#include <stdio.h>

#include "engines/gSLIC_core_engine.h"
//#include "engines/gSLIC_seg_engine_shared.h"

#include "gSLIC_io_tools.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#ifdef _DEBUG
#pragma  comment(lib, "opencv_world300d.lib")
#else
#pragma  comment(lib, "opencv_world300.lib")
#endif


using namespace std;
using namespace cv;

void load_image(const Mat& inimg, gSLIC::UChar4Image* outimg)
{
	gSLIC::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

void load_image(const gSLIC::UChar4Image* inimg, Mat& outimg)
{
	const gSLIC::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}


void main()
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cerr << "no video cap!\n";
		return;
	}
	
	gSLIC::objects::settings my_settings;

	my_settings.img_size.x = 640;
	my_settings.img_size.y = 480;
	my_settings.no_segs = 2000;
	my_settings.spixel_size = 16;
	my_settings.coh_weight = 1.0f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLIC::CIELAB;
	my_settings.seg_method = gSLIC::GIVEN_SIZE;

	my_settings.useGPU = true;

	gSLIC::engines::core_engine* gSLIC_engine = new gSLIC::engines::core_engine(my_settings);

	gSLIC::UChar4Image* in_img = new gSLIC::UChar4Image(my_settings.img_size, true, true);
	gSLIC::UChar4Image* out_img = new gSLIC::UChar4Image(my_settings.img_size, true, true);

	Mat oldFrame;
	Mat frame;
	Size s(my_settings.img_size.x, my_settings.img_size.y);

	unsigned char* imgBuffer = (unsigned char*)malloc(frame.cols*frame.rows*sizeof(unsigned char) * 4);
	memset(imgBuffer, 0, frame.cols*frame.rows * sizeof(unsigned char) * 4);

	while (cap.read(oldFrame))
	{
		resize(oldFrame, frame, s);
		
		load_image(frame, in_img);

		gSLIC_engine->Process_Frame(in_img);
		gSLIC_engine->Draw_Segmentation_Result(out_img);

		load_image(out_img, frame);

		imshow("frame", frame);

		if( waitKey(1) == 27 )break;
	}

	destroyAllWindows();
}