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

__global__ void Find_Center_Association_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size, float weight);
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

	Init_Cluster_Centers_device << <gridSize, blockSize >> >(img_ptr, spixel_list, map_size, img_size, spixel_size);
}

void gSLIC::engines::seg_engine_GPU::Find_Center_Association()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
	Vector4f* img_ptr = cvt_img->GetData(MEMORYDEVICE_CUDA);

	Vector2i map_size = spixel_map->noDims;
	Vector2i img_size = cvt_img->noDims;

	dim3 blockSize(spixel_size, spixel_size);
	dim3 gridSize(map_size.x,map_size.y);
	
	Find_Center_Association_device<< <gridSize, blockSize >> >(img_ptr, spixel_list, map_size, img_size, spixel_size, gslic_settings.coh_weight);
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

__global__ void Find_Center_Association_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size, float weight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	// load center information into shared memory
	__shared__ spixel_info shared_spixel_info[3][3];
	__shared__ bool shared_valid_mask[3][3];

	if (threadIdx.x<3 && threadIdx.y<3)
	{
		int ct_x = blockIdx.x + threadIdx.x - 1;
		int ct_y = blockIdx.y + threadIdx.y - 1;

		if (ct_x >=0 && ct_y >=0 && ct_x < map_size.x && ct_y < map_size.y)
		{
			int ct_idx = ct_y*map_size.x + ct_x;
			shared_spixel_info[threadIdx.x][threadIdx.y] = out_spixel[ct_idx];
			shared_valid_mask[threadIdx.x][threadIdx.y] = true;
		}
		else
		{
			shared_valid_mask[threadIdx.x][threadIdx.y] = false;
		}
	}
	__syncthreads();

	int idx_img = y * img_size.x + x;
	bool minidx = -1;
	float dist = 999999.9999f;
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
	{
		if (shared_valid_mask[i][j])
		{
			float cdist = compute_slic_distance(inimg[idx_img], x, y, shared_spixel_info[i][j], weight);
			if (cdist<dist)
			{
				dist = cdist;
				minidx = shared_spixel_info[i][j].id;
			}
		}
	}
	
	if (minidx>=0) 
}
