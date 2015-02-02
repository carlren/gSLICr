#ifndef __CUDA_SEG_SLIC__
#define  __CUDA_SEG_SLIC__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaDefines.h"

typedef struct
{
	float4 lab;
	float2 xy;
	int nPoints;

}SLICClusterCenter;

__host__ void SLICImgSeg(int* maskBuffer, float4* floatBuffer, 
						 int nWidth, int nHeight, int nSegs,  
						 SLICClusterCenter* vSLICCenterList, 
						 float weight);

__global__ void kInitClusterCenters(float4* floatBuffer, 
									int nWidth, int nHeight, int nSegs,  
									SLICClusterCenter* vSLICCenterList);

__global__ void kIterateKmeans(int* maskBuffer, float4* floatBuffer, 
							   int nWidth, int nHeight, int nSegs, int nClusterIdxStride, 
							   SLICClusterCenter* vSLICCenterList, 
							   bool bLabelImg, float weight);

__global__ void kUpdateClusterCenters(float4* floatBuffer, int* maskBuffer,
										  int nWidth, int nHeight, int nSegs,  
										  SLICClusterCenter* vSLICCenterList);

void enforceConnectivity(int* maskBuffer,int width, int height, int nSeg);

#endif
