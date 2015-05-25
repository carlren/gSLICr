#pragma once
#include "../gSLIC_defines.h"

namespace gSLIC
{
	namespace objects
	{
		class settings
		{
		public:
			Vector2i img_size;
			int no_segs;
			int spixel_size;
			int no_iters;
			float coh_weight;
			
			bool useGPU;

			COLOR_SPACE color_space;
			SEG_METHOD seg_method;

			settings();
			~settings(){};
		};

		settings::settings()
		{
			img_size.x			= 640;
			img_size.y			= 480;
			no_segs				= 2000;
			spixel_size			= 16;
			coh_weight			= 0.5f;
			no_iters			= 5;
			color_space			= XYZ;
			seg_method			= GIVEN_SIZE;

			useGPU				= true;
		}
	}
}