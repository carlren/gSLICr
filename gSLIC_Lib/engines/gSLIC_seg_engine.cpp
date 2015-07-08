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
	source_img->SetFrom(in_img, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CUDA);
	Cvt_Img_Space(source_img, cvt_img, gslic_settings.color_space);

	Init_Cluster_Centers();
	Find_Center_Association();

	for (int i = 0; i < gslic_settings.no_iters; i++)
	{
		Update_Cluster_Center();
		Find_Center_Association();
	}

	if(gslic_settings.do_enforce_connectivity) Enforce_Connectivity();
	cudaThreadSynchronize();
}



