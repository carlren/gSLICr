#include "cudaImgTrans.h"
#include "cudaUtil.h"


__host__ void Rgb2CIELab( uchar4* inputImg, float4* outputImg, int width, int height )
{
	dim3 ThreadPerBlock(BLCK_SIZE,BLCK_SIZE);
	dim3 BlockPerGrid(iDivUp(width,BLCK_SIZE),iDivUp(height,BLCK_SIZE));
	kRgb2CIELab<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);

}

__global__ void kRgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height)
{
	int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

	uchar4 nPixel=inputImg[offset];
	
	float _b=(float)nPixel.x/255.0;
	float _g=(float)nPixel.y/255.0;
	float _r=(float)nPixel.z/255.0;

	float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
	float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
	float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

	x/=0.950456;
	float y3=exp(log(y)/3.0);
	z/=1.088754;

	float l,a,b;

	x = x>0.008856 ? exp(log(x)/3.0) : (7.787*x+0.13793);
	y = y>0.008856 ? y3 : 7.787*y+0.13793;
	z = z>0.008856 ? z/=exp(log(z)/3.0) : (7.787*z+0.13793);
	
	l = y>0.008856 ? (116.0*y3-16.0) : 903.3*y;
	a=(x-y)*500.0;
	b=(y-z)*200.0;
	
	float4 fPixel;
	fPixel.x=l;
	fPixel.y=a;
	fPixel.z=b;

	outputImg[offset]=fPixel;
}


__host__ void Rgb2XYZ( uchar4* inputImg, float4* outputImg, int width, int height )
{
	dim3 ThreadPerBlock(BLCK_SIZE,BLCK_SIZE);
	dim3 BlockPerGrid(iDivUp(width,BLCK_SIZE),iDivUp(height,BLCK_SIZE));
	kRgb2XYZ<<<BlockPerGrid,ThreadPerBlock>>>(inputImg,outputImg,width,height);
}

__global__ void kRgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height)
{
	int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
	int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

	uchar4 nPixel=inputImg[offset];

	float _b=(float)nPixel.x/255.0;
	float _g=(float)nPixel.y/255.0;
	float _r=(float)nPixel.z/255.0;

	float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
	float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
	float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

	float4 fPixel;
	fPixel.x=x;
	fPixel.y=y;
	fPixel.z=z;

	outputImg[offset]=fPixel;
}

