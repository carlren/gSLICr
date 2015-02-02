#include "cudaSegEngine.h"
#include "cudaUtil.h"
#include "cudaImgTrans.h"
#include "cudaSegSLIC.h"
#include <time.h>

#include <stdio.h>
#include <math.h>
#include <time.h>

using namespace std;

__device__ uchar4* rgbBuffer;
__device__ float4* floatBuffer;
__device__ int* maskBuffer;
__device__ int* cMaskBuffer;

int nWidth,nHeight,nSeg,nMaxSegs;
bool cudaIsInitialized=false;


// for SLIC segmentation
int nClusterSize;
int nClustersPerCol;
int nClustersPerRow;
int nBlocksPerCluster;
int nBlocks;

int nBlockWidth;
int nBlockHeight;

__device__ SLICClusterCenter* vSLICCenterList;
bool slicIsInitialized=false;

__host__ void InitCUDA(int width, int height,int nSegment, SEGMETHOD eMethod)
{
	//for all methods
	if (!cudaIsInitialized)
	{
		nWidth=width;
		nHeight=height;

		cudaMalloc((void**) &rgbBuffer,width*height*sizeof(uchar4));
		cudaMalloc((void**) &floatBuffer,width*height*sizeof(float4));
		cudaMalloc((void**) &maskBuffer,width*height*sizeof(int));	

		cudaMemset(floatBuffer,0,width*height*sizeof(float4));
		cudaMemset(maskBuffer,0,width*height*sizeof(int));

		nSeg=nSegment;
		cudaIsInitialized=true;
	}

	switch(eMethod)
	{
	case SLIC:
		if (!slicIsInitialized)
		{
			nClusterSize=(int)sqrt((float)iDivUp(nWidth*nHeight,nSeg));

			nClustersPerCol=iDivUp(nHeight,nClusterSize);
			nClustersPerRow=iDivUp(nWidth,nClusterSize);
			nBlocksPerCluster=iDivUp(nClusterSize*nClusterSize,MAX_BLOCK_SIZE);
			nSeg=nClustersPerCol*nClustersPerRow;
			nMaxSegs=iDivUp(nWidth,BLCK_SIZE)*iDivUp(nHeight,BLCK_SIZE);
			nBlocks=nSeg*nBlocksPerCluster;

			nBlockWidth=nClusterSize;
			nBlockHeight=iDivUp(nClusterSize,nBlocksPerCluster);

			// the actual number of segments
			cudaMalloc((void**) &vSLICCenterList,nMaxSegs*sizeof(SLICClusterCenter));
			cudaMemset(vSLICCenterList,0,nMaxSegs*sizeof(SLICClusterCenter));
			slicIsInitialized=true;
		}
		break;
	}

}

extern "C" __host__ void CUDALoadImg(unsigned char* imgPixels)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(rgbBuffer,imgPixels,nWidth*nHeight*sizeof(uchar4),cudaMemcpyHostToDevice);
	}
	else
	{
		return;
	}
}

__host__ void TerminateCUDA()
{
	if (cudaIsInitialized)
	{
		cudaFree(rgbBuffer);
		cudaFree(floatBuffer);
		cudaFree(maskBuffer);
		cudaIsInitialized=false;
	}

	if (slicIsInitialized)
	{
		cudaFree(vSLICCenterList);
		slicIsInitialized=false;
	}

}

__host__ void CudaSegmentation( int nSegments, SEGMETHOD eSegmethod, double weight)
{
	nSeg=nSegments;

	switch (eSegmethod)
	{
	case SLIC :

		Rgb2CIELab(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,	nWidth,nHeight,nSeg,	vSLICCenterList,(float)weight);

		break;

	case RGB_SLIC:

		Uchar4ToFloat4(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,	nWidth,nHeight,nSeg,	vSLICCenterList,(float)weight);

		break;

	case XYZ_SLIC:

		Rgb2XYZ(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,vSLICCenterList,(float)weight);

		break;	
	}
	cudaThreadSynchronize();
}

__host__ void CopyImgDeviceToHost( unsigned char* imgPixels, int width, int height)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(imgPixels,rgbBuffer,nHeight*nWidth*sizeof(uchar4),cudaMemcpyDeviceToHost);
	}
}

__host__ void CopyMaskDeviceToHost( int* maskPixels, int width, int height)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(maskPixels,maskBuffer,nHeight*nWidth*sizeof(int),cudaMemcpyDeviceToHost);
	}
}

