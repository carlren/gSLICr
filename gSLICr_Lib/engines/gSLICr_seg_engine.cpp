// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine.h"

#include "../../NVTimer.h"
#include <iostream>

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;


seg_engine::seg_engine(const objects::settings& in_settings)
{
	gSLICr_settings = in_settings;
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
	Cvt_Img_Space(source_img, cvt_img, gSLICr_settings.color_space);

    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
    
	Init_Cluster_Centers();
	Find_Center_Association();

	for (int i = 0; i < gSLICr_settings.no_iters; i++)
	{
         sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
		Update_Cluster_Center();
        cudaThreadSynchronize();
        sdkStopTimer(&my_timer);
        cout<<"\rupdate cluster center in:["<<sdkGetTimerValue(&my_timer)<<"]ms";
        sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
		Find_Center_Association();
        cudaThreadSynchronize();
         cout<<"\tfind association in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
	}

	if(gSLICr_settings.do_enforce_connectivity) Enforce_Connectivity();
	cudaThreadSynchronize();
}



