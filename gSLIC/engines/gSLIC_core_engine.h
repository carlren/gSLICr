#pragma once
#include "gSLIC_seg_engine.h"

namespace gSLIC
{
	namespace engines
	{
		class core_engine
		{
		private:

			seg_engine* slic_seg_engine;

		public:

			core_engine(const objects::settings& in_settings)
			{
				if (in_settings.useGPU)
				{
					slic_seg_engine = new seg_engine(in_settings);
				}
			}

			~core_engine()
			{
				delete slic_seg_engine;
			}

			void Process_Frame(UChar4Image* in_img)
			{
				slic_seg_engine->Perform_Segmentation(in_img);
			}

			const IntImage * Get_Seg_Res()
			{
				return slic_seg_engine->Get_Seg_Mask();
			}

			void Draw_Segmentation_Result(UChar4Image* out_img)
			{
				slic_seg_engine->Draw_Segmentation_Result(out_img);
			};
		};
	}
}

