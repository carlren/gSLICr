#include "cudaSegSLIC.h"
#include "cudaUtil.h"
#include <stdio.h>

//index for enforce connectivity
const int dx4[4] = {-1,  0,  1,  0};
const int dy4[4] = { 0, -1,  0,  1};

__host__ void SLICImgSeg(int* maskBuffer, float4* floatBuffer, 
						 int nWidth, int nHeight, int nSegs,  
						 SLICClusterCenter* vSLICCenterList, 
						 float weight)
{
	int nClusterSize=(int)sqrt((float)iDivUp(nWidth*nHeight,nSegs));
	int nClustersPerCol=iDivUp(nHeight,nClusterSize);
	int nClustersPerRow=iDivUp(nWidth,nClusterSize);
	int nBlocksPerCluster=iDivUp(nClusterSize*nClusterSize,MAX_BLOCK_SIZE);

	int nSeg=nClustersPerCol*nClustersPerRow;

	int nBlockWidth=nClusterSize;
	int nBlockHeight=iDivUp(nClusterSize,nBlocksPerCluster);
	
	dim3 ThreadPerBlock_init(nClustersPerRow); //x
	dim3 BlockPerGrid_init(nClustersPerCol); //y
	
	dim3 ThreadPerBlock(nBlockWidth,nBlockHeight);
	dim3 BlockPerGrid(nBlocksPerCluster,nSeg);

	kInitClusterCenters<<<BlockPerGrid_init,ThreadPerBlock_init>>>(floatBuffer,nWidth,nHeight,nSegs,vSLICCenterList);

	//5 iterations have already given good result
	for (int i=0;i<5;i++)
	{
		kIterateKmeans<<<BlockPerGrid,ThreadPerBlock>>>(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);
		kUpdateClusterCenters<<<BlockPerGrid_init,ThreadPerBlock_init>>>(floatBuffer,maskBuffer,nWidth,nHeight,nSeg,vSLICCenterList);
	}
		kIterateKmeans<<<BlockPerGrid,ThreadPerBlock>>>(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,nClustersPerRow,vSLICCenterList,true, weight);
	
}


__global__ void kInitClusterCenters( float4* floatBuffer, int nWidth, int nHeight, int nSegs, SLICClusterCenter* vSLICCenterList )
{

	int blockWidth=nWidth/blockDim.x;
	int blockHeight=nHeight/gridDim.x;

	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int offsetBlock = blockIdx.x * blockHeight * nWidth + threadIdx.x * blockWidth;

	float2 avXY;

	avXY.x=threadIdx.x*blockWidth + (float)blockWidth/2.0;
	avXY.y=blockIdx.x*blockHeight + (float)blockHeight/2.0;

	//use a single point to init center
	int offset=offsetBlock + blockHeight/2 * nWidth+ blockWidth/2 ;

	float4 fPixel=floatBuffer[offset];

	vSLICCenterList[clusterIdx].lab=fPixel;
	vSLICCenterList[clusterIdx].xy=avXY;
	vSLICCenterList[clusterIdx].nPoints=0;
	
}

__global__ void kIterateKmeans( int* maskBuffer, float4* floatBuffer, 
							   int nWidth, int nHeight, int nSegs,  int nClusterIdxStride, 
							   SLICClusterCenter* vSLICCenterList, 
							   bool bLabelImg, float weight)
{

	//for reading cluster centers
	__shared__ float4 fShareLab[3][3];
	__shared__ float2 fShareXY[3][3];

	//pixel index
	__shared__ SLICClusterCenter pixelUpdateList[MAX_BLOCK_SIZE];
	__shared__ float2 pixelUpdateIdx[MAX_BLOCK_SIZE];


	int clusterIdx=blockIdx.y;
	int blockCol=clusterIdx%nClusterIdxStride;
	int blockRow=clusterIdx/nClusterIdxStride;
	//int upperBlockHeight=blockDim.y*gridDim.x;
	
	int lowerBlockHeight=blockDim.y;
	int blockWidth=blockDim.x;
	int upperBlockHeight=blockWidth;

	int innerBlockHeightIdx=lowerBlockHeight*blockIdx.x+threadIdx.y;

	float M=weight;
	float invWeight=1/((blockWidth/M)*(blockWidth/M));

	int offsetBlock = (blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight)*nWidth+blockCol*blockWidth;
	int offset=offsetBlock+threadIdx.x+threadIdx.y*nWidth;

	int rBegin=(blockRow>0)?0:1;
	int rEnd=(blockRow+1>(gridDim.y/nClusterIdxStride-1))?1:2;
	int cBegin=(blockCol>0)?0:1;
	int cEnd=(blockCol+1>(nClusterIdxStride-1))?1:2;
	
	if (threadIdx.x<3 && threadIdx.y<3)
	{
		if (threadIdx.x>=cBegin && threadIdx.x<=cEnd && threadIdx.y>=rBegin && threadIdx.y<=rEnd)
		{
			int cmprIdx=(blockRow+threadIdx.y-1)*nClusterIdxStride+(blockCol+threadIdx.x-1);
			fShareLab[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].lab;
			fShareXY[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].xy;
		}
	}
	__syncthreads();

	if (innerBlockHeightIdx>=blockWidth)
	{
		return;
	}

	if (offset>=nWidth*nHeight)
	{
		return;
	}

	// finding the nearest center for current pixel
	float fY=blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight+threadIdx.y;
	float fX=blockCol*blockWidth+threadIdx.x;

	if (fY<nHeight && fX<nWidth)
	{
		float4 fPoint=floatBuffer[offset];
		float minDis=9999;
		int nearestCenter=-1;
		int nearestR, nearestC;

		for (int r=rBegin;r<=rEnd;r++)
		{
			for (int c=cBegin;c<=cEnd;c++)
			{
				int cmprIdx=(blockRow+r-1)*nClusterIdxStride+(blockCol+c-1);

				//compute SLIC distance
				float fDab=(fPoint.x-fShareLab[r][c].x)*(fPoint.x-fShareLab[r][c].x)
					+(fPoint.y-fShareLab[r][c].y)*(fPoint.y-fShareLab[r][c].y)
					+(fPoint.z-fShareLab[r][c].z)*(fPoint.z-fShareLab[r][c].z);
				//fDab=sqrt(fDab);

				float fDxy=(fX-fShareXY[r][c].x)*(fX-fShareXY[r][c].x)
					+(fY-fShareXY[r][c].y)*(fY-fShareXY[r][c].y);
				//fDxy=sqrt(fDxy);

				float fDis=fDab+invWeight*fDxy;

				if (fDis<minDis)
				{
					minDis=fDis;
					nearestCenter=cmprIdx;
					nearestR=r;
					nearestC=c;
				}

			}
		}

		if (nearestCenter>-1)
		{
			int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

			pixelUpdateList[pixelIdx].lab=fPoint;
			pixelUpdateList[pixelIdx].xy.x=fX;
			pixelUpdateList[pixelIdx].xy.y=fY;

			pixelUpdateIdx[pixelIdx].x=nearestC;
			pixelUpdateIdx[pixelIdx].y=nearestR;
			
			if (bLabelImg)
			{
				maskBuffer[offset]=nearestCenter;
			}
		}
	}
	else
	{
		int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

		pixelUpdateIdx[pixelIdx].x=-1;
		pixelUpdateIdx[pixelIdx].y=-1;

	}
	__syncthreads();
}

// from original SLIC
void FindNext(const int* labels, int* nlabels, const int& height, const int& width, const int& h,	const int& w,
					const int&	lab,	int*	xvec, int* yvec, int& count)
{
	int oldlab = labels[h*width+w];
	for( int i = 0; i < 4; i++ )
	{
		int y = h+dy4[i];int x = w+dx4[i];
		if((y < height && y >= 0) && (x < width && x >= 0) )
		{
			int ind = y*width+x;
			if(nlabels[ind] < 0 && labels[ind] == oldlab )
			{
				xvec[count] = x;
				yvec[count] = y;
				count++;
				nlabels[ind] = lab;
				FindNext(labels, nlabels, height, width, y, x, lab, xvec, yvec, count);
			}
		}
	}
}

__global__ void kUpdateClusterCenters( float4* floatBuffer,int* maskBuffer, int nWidth, int nHeight, int nSegs, SLICClusterCenter* vSLICCenterList )
{

	int blockWidth=nWidth/blockDim.x;
	int blockHeight=nHeight/gridDim.x;

	int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int offsetBlock = threadIdx.x * blockWidth+ blockIdx.x * blockHeight * nWidth;

	float2 crntXY=vSLICCenterList[clusterIdx].xy;
	float4 avLab;
	float2 avXY;
	int nPoints=0;

	avLab.x=0;
	avLab.y=0;
	avLab.z=0;

	avXY.x=0;
	avXY.y=0;

	int yBegin=0 < (crntXY.y - blockHeight) ? (crntXY.y - blockHeight) : 0;
	int yEnd= nHeight > (crntXY.y + blockHeight) ? (crntXY.y + blockHeight) : (nHeight-1);	
	int xBegin=0 < (crntXY.x - blockWidth) ? (crntXY.x - blockWidth) : 0;
	int xEnd= nWidth > (crntXY.x + blockWidth) ? (crntXY.x + blockWidth) : (nWidth-1);

	//update to cluster centers
	// 
	for (int i = yBegin; i < yEnd ; i++)
	{
		for (int j = xBegin; j < xEnd; j++)
		{
			int offset=j + i * nWidth;			
			float4 fPixel=floatBuffer[offset];
			int pIdx=maskBuffer[offset];

			if (pIdx==clusterIdx)
			{
				avLab.x+=fPixel.x;
				avLab.y+=fPixel.y;
				avLab.z+=fPixel.z;

				avXY.x+=j;
				avXY.y+=i;

				nPoints++;
			}
		}
	}

	avLab.x/=nPoints;
	avLab.y/=nPoints;
	avLab.z/=nPoints;

	avXY.x/=nPoints;
	avXY.y/=nPoints;

	vSLICCenterList[clusterIdx].lab=avLab;
	vSLICCenterList[clusterIdx].xy=avXY;
	vSLICCenterList[clusterIdx].nPoints=nPoints;
}



void enforceConnectivity(int* maskBuffer,int width, int height, int nSeg)
{
	int sz = width*height;
	int* nlabels=(int*)malloc(sz*sizeof(int));
	memset(nlabels,-1,sz*sizeof(int));
	int* labels=maskBuffer;

	const int SUPSZ = sz/nSeg;

	//------------------
	// labeling
	//------------------
	int lab=0;
	int i=0;
	int adjlabel=0;//adjacent label
	int* xvec = (int*)malloc(sz*sizeof(int)); //worst case size
	int* yvec = (int*)malloc(sz*sizeof(int)); //worst case size
	{for( int h = 0; h < height; h++ )
	{
		for( int w = 0; w < width; w++ )
		{
			if(nlabels[i] < 0)
			{
				nlabels[i] = lab;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = w + dx4[n];
					int y = h + dy4[n];

					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}
				xvec[0] = w; yvec[0] = h;
				int count=1;
				FindNext(labels, nlabels, height, width, h, w, lab, xvec, yvec, count);
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= (SUPSZ >> 2))
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					lab--;
				}
				lab++;
			}
			i++;
		}
	}}
	//------------------
	//numlabels = lab;
	//------------------
	if(xvec) free(xvec);
	if(yvec) free(yvec);

	memcpy(labels,nlabels,sz*sizeof(int));
	if (nlabels)
	{
		free(nlabels);
	}
	
	cudaThreadSynchronize();
}

