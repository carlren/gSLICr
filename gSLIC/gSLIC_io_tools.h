#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "gSLIC_defines.h"

template <typename T>
inline void WriteMatlabTXTImg(char* fileName, T *imgData, int w, int h)
{
	std::ofstream fid;
	fid.open(fileName);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int idx = i*w + j;
			fid << imgData[idx] << '\t';
		}
		fid << '\n';
	}
	fid.close();
}


static inline void WritePPMimage(char* fileName, gSLIC::Vector4u *imgData, int w, int h)
{
	gSLIC::Vector3u *tmpData = new gSLIC::Vector3u[w*h];
	for (int i = 0; i < w*h; i++) tmpData[i] = imgData[i].toVector3();


	FILE* fid = fopen(fileName, "w");
	fprintf(fid, "P6\n");
	fprintf(fid, "%d %d\n", w, h);
	fprintf(fid, "255\n");
	fclose(fid);

	fid = fopen(fileName, "ab+");
	fwrite(tmpData, sizeof(char), w*h * 3, fid);
	fclose(fid);
}

