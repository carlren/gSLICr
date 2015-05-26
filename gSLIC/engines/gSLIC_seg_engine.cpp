#pragma once
#include "gSLIC_seg_engine.h"

#include "../gSLIC_io_tools.h"

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
	source_img->SetFrom(in_img, UChar4Image::MemoryCopyDirection::CPU_TO_CUDA);
	Cvt_Img_Space(source_img, cvt_img, gslic_settings.color_space);

	Init_Cluster_Centers();

	for (int i = 0; i < gslic_settings.no_iters; i++)
	{
		Find_Center_Association();
		Update_Cluster_Center();
	}
	Find_Center_Association();
	//Enforce_Connectivity();
	cudaThreadSynchronize();
}

void seg_engine::Enforce_Connectivity()
{

}



