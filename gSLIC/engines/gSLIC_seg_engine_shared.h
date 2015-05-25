#pragma once
#include "../gSLIC_defines.h"
#include "../objects/gSLIC_spixel_info.h"

_CPU_AND_GPU_CODE_ inline void rgb2xyz(const gSLIC::Vector4u& pix_in, gSLIC::Vector4f& pix_out)
{
	float _b = (float)pix_in.x / 255.0;
	float _g = (float)pix_in.y / 255.0;
	float _r = (float)pix_in.z / 255.0;

	pix_out.x = _r*0.412453 + _g*0.357580 + _b*0.180423;
	pix_out.y = _r*0.212671 + _g*0.715160 + _b*0.072169;
	pix_out.z = _r*0.019334 + _g*0.119193 + _b*0.950227;

}

_CPU_AND_GPU_CODE_ inline void rgb2CIELab(const gSLIC::Vector4u& pix_in, gSLIC::Vector4f& pix_out)
{
	float _b = (float)pix_in.x / 255.0;
	float _g = (float)pix_in.y / 255.0;
	float _r = (float)pix_in.z / 255.0;

	float x = _r*0.412453 + _g*0.357580 + _b*0.180423;
	float y = _r*0.212671 + _g*0.715160 + _b*0.072169;
	float z = _r*0.019334 + _g*0.119193 + _b*0.950227;

	x /= 0.950456;
	float y3 = exp(log(y) / 3.0);
	z /= 1.088754;

	x = x > 0.008856 ? exp(log(x) / 3.0) : (7.787*x + 0.13793);
	y = y > 0.008856 ? y3 : 7.787*y + 0.13793;
	z = z > 0.008856 ? z /= exp(log(z) / 3.0) : (7.787*z + 0.13793);

	pix_out.x = y > 0.008856 ? (116.0*y3 - 16.0) : 903.3*y; // l
	pix_out.y = (x - y)*500.0; // a
	pix_out.z = (y - z)*200.0; // b
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
	
	out_spixel[cluster_idx].accum_center = gSLIC::Vector2f(0, 0);
	out_spixel[cluster_idx].accum_color_info = gSLIC::Vector4f(0,0,0,0);
	out_spixel[cluster_idx].no_pixels = 0;
}

_CPU_AND_GPU_CODE_ inline float compute_slic_distance(const gSLIC::Vector4f& pix, int x, int y, const gSLIC::objects::spixel_info& center_info, float weight)
{
	float dcolor = (pix.x - center_info.color_info.x)*(pix.x - center_info.color_info.x)
				 + (pix.y - center_info.color_info.y)*(pix.y - center_info.color_info.y)
				 + (pix.z - center_info.color_info.z)*(pix.z - center_info.color_info.z);
	dcolor = sqrtf(dcolor);

	float dxy = (x - center_info.center.x) * (x - center_info.center.x)
			  + (y - center_info.center.y) * (y - center_info.center.y);
	dxy = sqrtf(dxy);

	return dcolor + weight * dxy;
}

_CPU_AND_GPU_CODE_ inline void find_center_association_shared(const gSLIC::Vector4f* inimg, const gSLIC::objects::spixel_info* in_spixel_map, int* out_idx_img, gSLIC::Vector2i map_size, gSLIC::Vector2i img_size, int spixel_size, float weight, int x, int y)
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
			float cdist = compute_slic_distance(inimg[idx_img], x, y, in_spixel_map[ctr_idx], weight);
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
		outimg[idx] = gSLIC::Vector4u(255,0,0,0);
	}
	else
	{
		outimg[idx] = sourceimg[idx];
	}
}
