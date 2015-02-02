#ifndef __CUDA_DEFINES__
#define __CUDA_DEFINES__

#define BLCK_SIZE 16
#define MAX_BLOCK_SIZE 256

typedef enum _SEGMETHOD
{
	SLIC = 0,
	RGB_SLIC,
	XYZ_SLIC,

} SEGMETHOD;

#endif