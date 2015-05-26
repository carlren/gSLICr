#pragma once
#include "gSLIC_seg_engine_GPU.h"
#include "../NVTimer.h"

namespace gSLIC
{
	namespace engines
	{
		class core_engine
		{
		private:

			seg_engine* slic_seg_engine;
			StopWatchInterface *slic_timer;

		public:

			core_engine(const objects::settings& in_settings)
			{
				if (in_settings.useGPU)
				{
					slic_seg_engine = new seg_engine_GPU(in_settings);
					sdkCreateTimer(&slic_timer);
				}
			}

			~core_engine()
			{
				delete slic_seg_engine;
				delete slic_timer;
			}

			void Process_Frame(UChar4Image* in_img)
			{
				sdkResetTimer(&slic_timer); sdkStartTimer(&slic_timer);
				slic_seg_engine->Perform_Segmentation(in_img);
				sdkStopTimer(&slic_timer); printf("\rSegmentation in:[%.2f]ms ", sdkGetTimerValue(&slic_timer));
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

