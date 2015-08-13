// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "../gSLICr_defines.h"

namespace gSLICr
{
	namespace objects
	{
		struct settings
		{
			Vector2i img_size;
			int no_segs;
			int spixel_size;
			int no_iters;
			float coh_weight;			
			bool do_enforce_connectivity;

			COLOR_SPACE color_space;
			SEG_METHOD seg_method;
		};
	}
}