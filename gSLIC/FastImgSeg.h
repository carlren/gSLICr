#ifndef __CUDA_SUPERPIXELSEG__
#define __CUDA_SUPERPIXELSEG__

#include "cudaUtil.h"
#include "cudaSegSLIC.h"

class FastImgSeg
{

public:
	unsigned char* sourceImage;
	unsigned char* markedImg;
	int* segMask;

private:

	int width;
	int height;
	int nSeg;

	bool bSegmented;
	bool bImgLoaded;
	bool bMaskGot;

public:
	FastImgSeg();
	FastImgSeg(int width,int height,int dim,int nSegments);
	~FastImgSeg();

	void initializeFastSeg(int width,int height,int nSegments);
	void clearFastSeg();
	void changeClusterNum(int nSegments);

	void LoadImg(unsigned char* imgP);
	void DoSegmentation(SEGMETHOD eMethod, double weight);
	void Tool_GetMarkedImg();
	void Tool_WriteMask2File(char* outFileName, bool writeBinary);
};

#endif
