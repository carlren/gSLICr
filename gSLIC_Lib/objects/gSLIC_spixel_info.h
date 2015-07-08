#pragma once
#include "../gSLIC_defines.h"

namespace gSLIC
{
	namespace objects
	{
		struct spixel_info
		{
			Vector2f center;
			Vector4f color_info;
			int id;
			int no_pixels;
		};
	}

	typedef ORUtils::Image<objects::spixel_info> SpixelMap;
}