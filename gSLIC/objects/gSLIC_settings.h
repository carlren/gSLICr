#pragma once
#include "../gSLIC_defines.h"

namespace gSLIC
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
			
			bool enforce_connectivity;

			bool useGPU; // to be removed

			COLOR_SPACE color_space;
			SEG_METHOD seg_method;
		};
	}
}