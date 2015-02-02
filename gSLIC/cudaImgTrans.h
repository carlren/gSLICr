#ifndef __CUDA_IMG_TRANS__
#define __CUDA_IMG_TRANS__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaDefines.h"

__host__ void Rgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height);

__host__ void Rgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height);

#endif