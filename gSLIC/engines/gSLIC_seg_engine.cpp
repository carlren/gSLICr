#pragma once
#include "gSLIC_seg_engine.h"

using namespace std;
using namespace gSLIC;
using namespace gSLIC::objects;
using namespace gSLIC::engines;


seg_engine::seg_engine(const objects::settings& in_settings)
{
	gslic_settings = in_settings;
}


seg_engine::~seg_engine()
{
	if (source_img != NULL) delete source_img;
	if (cvt_img != NULL) delete cvt_img;
	if (idx_img != NULL) delete idx_img;
	if (spixel_map != NULL) delete spixel_map;
}

void seg_engine::Perform_Segmentation(UChar4Image* in_img)
{
	has_segmented = false;
	has_img_loaded = false;
}

void seg_engine::Perform_Segmentation(UChar4Image* in_img)
{
	Init_Cluster_Centers();
	for (int i = 0; i < gslic_settings.no_iters; i++)
	{
		Find_Center_Association();
		Update_Cluster_Center();
	}
	Find_Center_Association();
	Enforce_Connectivity();
}

void seg_engine::Enforce_Connectivity()
{

}


//void FastImgSeg::Tool_GetMarkedImg()
//{
//	if (!bSegmented)
//	{
//		return;
//	}
//
//	memcpy(markedImg,sourceImage,width*height*4*sizeof(unsigned char));
//
//	for (int i=1;i<height-1;i++)
//	{
//		for (int j=1;j<width-1;j++)
//		{
//			int mskIndex=i*width+j;
//			if (segMask[mskIndex]!=segMask[mskIndex+1] 
//			|| segMask[mskIndex]!=segMask[(i-1)*width+j]
//			|| segMask[mskIndex]!=segMask[mskIndex-1]
//			|| segMask[mskIndex]!=segMask[(i+1)*width+j])
//			{
//				markedImg[mskIndex*4]=0;
//				markedImg[mskIndex*4+1]=0;
//				markedImg[mskIndex*4+2]=255;
//			}
//		}
//	}
//
//}


