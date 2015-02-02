#ifndef __CUDA_SEG_ENGINE__
#define __CUDA_SEG_ENGINE__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaDefines.h"

extern "C" __host__ void InitCUDA(int width, int height,int nSegment, SEGMETHOD eMethod);
extern "C" __host__ void CUDALoadImg(unsigned char* imgPixels);

extern "C" __host__ void TerminateCUDA();
extern "C" __host__ void CopyImgDeviceToHost(unsigned char* imgPixels, int width, int height);
extern "C" __host__ void CopyMaskDeviceToHost(int* maskPixels, int width, int height);
extern "C" __host__ void CudaSegmentation(int nSegments, SEGMETHOD eSegmethod, double weight);


#endif