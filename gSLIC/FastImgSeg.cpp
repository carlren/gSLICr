#include "FastImgSeg.h"
#include "cudaSegEngine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

FastImgSeg::FastImgSeg(int w,int h, int d, int nSegments)
{
	width=w;
	height=h;
	nSeg=nSegments;

	segMask=(int*) malloc(width*height*sizeof(int));
	markedImg=(unsigned char*)malloc(width*height*4*sizeof(unsigned char));

	InitCUDA(width,height,nSegments,SLIC);

	bImgLoaded=false;
	bSegmented=false;
}

FastImgSeg::FastImgSeg()
{

}

FastImgSeg::~FastImgSeg()
{
	clearFastSeg();
}


void FastImgSeg::changeClusterNum(int nSegments)
{
	nSeg=nSegments;
}

void FastImgSeg::initializeFastSeg(int w,int h, int nSegments)
{
	width=w;
	height=h;
	nSeg=nSegments;

	segMask=(int*) malloc(width*height*sizeof(int));
	markedImg=(unsigned char*)malloc(width*height*4*sizeof(unsigned char));

	InitCUDA(width,height,nSegments,SLIC);

	bImgLoaded=false;
	bSegmented=false;
}

void FastImgSeg::clearFastSeg()
{
	free(segMask);
	free(markedImg);
	TerminateCUDA();
	bImgLoaded=false;
	bSegmented=false;
}


void FastImgSeg::LoadImg(unsigned char* imgP)
{
	sourceImage=imgP;
	CUDALoadImg(sourceImage);
	bSegmented=false;
}

void FastImgSeg::DoSegmentation(SEGMETHOD eMethod, double weight)
{
		clock_t start,finish;

		start=clock();
		CudaSegmentation(nSeg,eMethod, weight);
		finish=clock();
		printf("clustering:%f\t",(double)(finish-start)/CLOCKS_PER_SEC);

		CopyMaskDeviceToHost(segMask,width,height);

		start=clock();
		enforceConnectivity(segMask,width,height,nSeg);
		finish=clock();
		printf("connectivity:%f\n",(double)(finish-start)/CLOCKS_PER_SEC);

		bSegmented=true;
}

void FastImgSeg::Tool_GetMarkedImg()
{
	if (!bSegmented)
	{
		return;
	}

	memcpy(markedImg,sourceImage,width*height*4*sizeof(unsigned char));

	for (int i=1;i<height-1;i++)
	{
		for (int j=1;j<width-1;j++)
		{
			int mskIndex=i*width+j;
			if (segMask[mskIndex]!=segMask[mskIndex+1] 
			|| segMask[mskIndex]!=segMask[(i-1)*width+j]
			|| segMask[mskIndex]!=segMask[mskIndex-1]
			|| segMask[mskIndex]!=segMask[(i+1)*width+j])
			{
				markedImg[mskIndex*4]=0;
				markedImg[mskIndex*4+1]=0;
				markedImg[mskIndex*4+2]=255;
			}
		}
	}

}


void FastImgSeg::Tool_WriteMask2File(char* outFileName, bool writeBinary)
{
	if (!bSegmented)
	{
		return;
	}
	
	if (writeBinary)
	{
		ofstream outf;
		outf.open(outFileName, ios::binary);

		outf.write(reinterpret_cast<char*>(&width),sizeof(width));
		outf.write(reinterpret_cast<char*>(&height),sizeof(height));

		for (int i=0;i<height;i++)
		{
			for (int j=0;j<width;j++)
			{
				int mskIndex=i*width+j;
				int idx=segMask[mskIndex];
				outf.write(reinterpret_cast<char*>(&idx),sizeof(idx));
			}
		}
		outf.close();
	}
	else
	{
		ofstream outf;

		outf.open(outFileName);

		for (int i=0;i<height;i++)
		{
			for (int j=0;j<width;j++)
			{
				int mskIndex=i*width+j;
				int idx=segMask[mskIndex];
				outf<<idx<<' ';
			}
			outf<<'\n';
		}
		outf.close();
	}
}