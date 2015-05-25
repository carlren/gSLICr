#include "gSLIC_seg_engine_GPU.h"
#include "gSLIC_seg_engine_shared.h"

using namespace std;
using namespace gSLIC;
using namespace gSLIC::objects;
using namespace gSLIC::engines;


// ----------------------------------------------------
//
//	kernel function defines
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const Vector4u* inimg, Vector4f* outimg, Vector2i img_size, COLOR_SPACE color_space);

__global__ void Init_Cluster_Centers_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size);

__global__ void Find_Center_Association_device();
// ----------------------------------------------------
//
//	host function implementations
//
// ----------------------------------------------------

seg_engine_GPU::seg_engine_GPU(const settings& in_settings) : seg_engine(in_settings)
{
	source_img = new UChar4Image(in_settings.img_size,true,true);
	cvt_img = new Float4Image(in_settings.img_size, true, true);
	idx_img = new IntImage(in_settings.img_size, true, true);

	if (in_settings.seg_method == GIVEN_NUM)
	{
		float cluster_size = (float)(in_settings.img_size.x * in_settings.img_size.x) / (float)in_settings.no_segs;
		spixel_size = ceil(sqrtf(cluster_size));
	}
	spixel_size = spixel_size > MAX_SPIXEL_SIZE ? MAX_SPIXEL_SIZE : spixel_size;

	int spixel_per_col = ceil(in_settings.img_size.x / spixel_size);
	int spixel_per_row = ceil(in_settings.img_size.y / spixel_size);
	
	spixel_map = new SpixelMap(Vector2i(spixel_per_col,spixel_per_row),true,true);
}

void gSLIC::engines::seg_engine_GPU::Cvt_Img_Space(UChar4Image* inimg, Float4Image* outimg, COLOR_SPACE color_space)
{
	Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CUDA);
	Vector4f* outimg_ptr = outimg->GetData(MEMORYDEVICE_CUDA);
	Vector2i img_size = inimg->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Cvt_Img_Space_device << <gridSize, blockSize >> >(inimg_ptr, outimg_ptr, img_size, color_space);

}

void gSLIC::engines::seg_engine_GPU::Init_Cluster_Centers()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
	Vector4f* img_ptr = cvt_img->GetData(MEMORYDEVICE_CUDA);

	Vector2i map_size = spixel_map->noDims;
	Vector2i img_size = cvt_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)map_size.x / (float)blockSize.x), (int)ceil((float)map_size.y / (float)blockSize.y));


}

void gSLIC::engines::seg_engine_GPU::Find_Center_Association()
{

}

void gSLIC::engines::seg_engine_GPU::Enforce_Connectivity()
{

}

// ----------------------------------------------------
//
//	device function implementations
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const Vector4u* inimg, Vector4f* outimg, Vector2i img_size, COLOR_SPACE color_space)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	cvt_img_space_shared(inimg, outimg, img_size, x, y, color_space);

}

__global__ void Init_Cluster_Centers_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > map_size.x - 1 || y > map_size.y - 1) return;

	init_cluster_centers_shared(inimg, out_spixel, map_size, img_size, spixel_size, x, y);
}
