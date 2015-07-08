#pragma once
#include "../gSLIC_defines.h"
#include "../objects/gSLIC_spixel_info.h"

_CPU_AND_GPU_CODE_ inline void rgb2xyz(const gSLIC::Vector4u& pix_in, gSLIC::Vector4f& pix_out)
{
	float _b = (float)pix_in.x * 0.0039216f;
	float _g = (float)pix_in.y * 0.0039216f;
	float _r = (float)pix_in.z * 0.0039216f;

	pix_out.x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
	pix_out.y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
	pix_out.z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

}

_CPU_AND_GPU_CODE_ inline void rgb2CIELab(const gSLIC::Vector4u& pix_in, gSLIC::Vector4f& pix_out)
{
	float _b = (float)pix_in.x * 0.0039216f;
	float _g = (float)pix_in.y * 0.0039216f;
	float _r = (float)pix_in.z * 0.0039216f;

	float x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
	float y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
	float z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = x / Xr;
	double yr = y / Yr;
	double zr = z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa*zr + 16.0) / 116.0;

	pix_out.x = 116.0*fy - 16.0;
	pix_out.y = 500.0*(fx - fy);
	pix_out.z = 200.0*(fy - fz);
}

_CPU_AND_GPU_CODE_ inline void cvt_img_space_shared(const gSLIC::Vector4u* inimg, gSLIC::Vector4f* outimg, const gSLIC::Vector2i& img_size, int x, int y, const gSLIC::COLOR_SPACE& color_space)
{
	int idx = y * img_size.x + x;

	switch (color_space)
	{
	case gSLIC::RGB:
		outimg[idx].x = inimg[idx].x;
		outimg[idx].y = inimg[idx].y;
		outimg[idx].z = inimg[idx].z;
		break;
	case gSLIC::XYZ:
		rgb2xyz(inimg[idx], outimg[idx]);
		break;
	case gSLIC::CIELAB:
		rgb2CIELab(inimg[idx], outimg[idx]);
		break;
	}
}

_CPU_AND_GPU_CODE_ inline void init_cluster_centers_shared(const gSLIC::Vector4f* inimg, gSLIC::objects::spixel_info* out_spixel, gSLIC::Vector2i map_size, gSLIC::Vector2i img_size, int spixel_size, int x, int y)
{
	int cluster_idx = y * map_size.x + x;

	int img_x = x * spixel_size + spixel_size / 2;
	int img_y = y * spixel_size + spixel_size / 2;

	img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
	img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;

	// TODO: go one step towards gradients direction

	out_spixel[cluster_idx].id = cluster_idx;
	out_spixel[cluster_idx].center = gSLIC::Vector2f(img_x, img_y);
	out_spixel[cluster_idx].color_info = inimg[img_y*img_size.x + img_x];
	
	out_spixel[cluster_idx].no_pixels = 0;
}

_CPU_AND_GPU_CODE_ inline float compute_slic_distance(const gSLIC::Vector4f& pix, int x, int y, const gSLIC::objects::spixel_info& center_info, float weight, float normalizer_xy, float normalizer_color)
{
	float dcolor = (pix.x - center_info.color_info.x)*(pix.x - center_info.color_info.x)
				 + (pix.y - center_info.color_info.y)*(pix.y - center_info.color_info.y)
				 + (pix.z - center_info.color_info.z)*(pix.z - center_info.color_info.z);

	float dxy = (x - center_info.center.x) * (x - center_info.center.x)
			  + (y - center_info.center.y) * (y - center_info.center.y);


	float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;
	return sqrtf(retval);
}

_CPU_AND_GPU_CODE_ inline void find_center_association_shared(const gSLIC::Vector4f* inimg, const gSLIC::objects::spixel_info* in_spixel_map, int* out_idx_img, gSLIC::Vector2i map_size, gSLIC::Vector2i img_size, int spixel_size, float weight, int x, int y, float max_xy_dist, float max_color_dist)
{
	int idx_img = y * img_size.x + x;

	int ctr_x = x / spixel_size;
	int ctr_y = y / spixel_size;

	int minidx = -1;
	float dist = 999999.9999f;

	// search 3x3 neighborhood
	for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
	{
		int ctr_x_check = ctr_x + j;
		int ctr_y_check = ctr_y + i;
		if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y)
		{
			int ctr_idx = ctr_y_check*map_size.x + ctr_x_check;
			float cdist = compute_slic_distance(inimg[idx_img], x, y, in_spixel_map[ctr_idx], weight, max_xy_dist, max_color_dist);
			if (cdist < dist)
			{
				dist = cdist;
				minidx = in_spixel_map[ctr_idx].id;
			}
		}
	}

	if (minidx >= 0) out_idx_img[idx_img] = minidx;
}

_CPU_AND_GPU_CODE_ inline void draw_superpixel_boundry_shared(const int* idx_img, gSLIC::Vector4u* sourceimg, gSLIC::Vector4u* outimg, gSLIC::Vector2i img_size, int x, int y)
{
	int idx = y * img_size.x + x;

	if (idx_img[idx] != idx_img[idx + 1]
	 || idx_img[idx] != idx_img[idx - 1]
	 || idx_img[idx] != idx_img[(y - 1)*img_size.x + x]
	 || idx_img[idx] != idx_img[(y + 1)*img_size.x + x])
	{
		outimg[idx] = gSLIC::Vector4u(0,0,255,0);
	}
	else
	{
		outimg[idx] = sourceimg[idx];
	}
}

_CPU_AND_GPU_CODE_ inline void finalize_reduction_result_shared(const gSLIC::objects::spixel_info* accum_map, gSLIC::objects::spixel_info* spixel_list, gSLIC::Vector2i map_size, int no_blocks_per_spixel, int x, int y)
{
	int spixel_idx = y * map_size.x + x;

	spixel_list[spixel_idx].center = gSLIC::Vector2f(0, 0);
	spixel_list[spixel_idx].color_info = gSLIC::Vector4f(0, 0, 0, 0);
	spixel_list[spixel_idx].no_pixels = 0;

	for (int i = 0; i < no_blocks_per_spixel; i++)
	{
		int accum_list_idx = spixel_idx * no_blocks_per_spixel + i;

		spixel_list[spixel_idx].center += accum_map[accum_list_idx].center;
		spixel_list[spixel_idx].color_info += accum_map[accum_list_idx].color_info;
		spixel_list[spixel_idx].no_pixels += accum_map[accum_list_idx].no_pixels;
	}

	if (spixel_list[spixel_idx].no_pixels != 0)
	{
		spixel_list[spixel_idx].center /= (float)spixel_list[spixel_idx].no_pixels;
		spixel_list[spixel_idx].color_info /= (float)spixel_list[spixel_idx].no_pixels;
	}
}

_CPU_AND_GPU_CODE_ inline void supress_local_lable(const int* in_idx_img, int* out_idx_img, gSLIC::Vector2i img_size, int x, int y)
{
	int clable = in_idx_img[y*img_size.x + x];

	// don't suppress boundary
	if (x <= 1 || y <= 1 || x >= img_size.x - 2 || y >= img_size.y - 2)
	{ 
		out_idx_img[y*img_size.x + x] = clable;
		return; 
	}

	int diff_count = 0;
	int diff_lable = -1;

	for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++)
	{
		int nlable = in_idx_img[(y + j)*img_size.x + (x + i)];
		if (nlable!=clable)
		{
			diff_lable = nlable;
			diff_count++;
		}
	}

	if (diff_count>=16)
		out_idx_img[y*img_size.x + x] = diff_lable;
	else
		out_idx_img[y*img_size.x + x] = clable;
}