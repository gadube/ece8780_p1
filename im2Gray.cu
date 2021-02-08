#include "im2Gray.h"


#ifndef BLOCK
#define BLOCK 32
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */
	size_t x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < numCols && y < numRows)
	{
		// gray pixel index
		int i = y * numCols + x;

		unsigned char r = d_in[i].x; // red pixel value
		unsigned char g = d_in[i].y; // green pixel value
		unsigned char b = d_in[i].z; // blue pixel value

		// grayscale conversion using formula 1 from project doc
		d_grey[i] = 0.299f * r + 0.587f * g + 0.114f * b;
	}
	return;
}


__global__ 
void im2Gray_s(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){

	__shared__ uchar4 ds_in[TILE_WIDTH][TILE_WIDTH];


	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	size_t row = by * blockDim.y + ty;
	size_t col = bx * blockDim.x + tx;
	int i = row * numCols + col;

	if(col < numCols && row < numRows){
//	for(int p =0; p < numCols/TILE_WIDTH; ++p){
//		if (col < p * TILE_WIDTH + tx && row < numRows) {
			//ds_in[ty][tx] = d_in[(p * TILE_WIDTH + ty) * numCols + col]; 
			int p = col / TILE_WIDTH;
			if (col < numCols && row < numRows){
			ds_in[ty][tx] =  d_in[row * numCols + col];
//		}
			__syncthreads();

	//	for(int j = 0; j < TILE_WIDTH; j++){
			

				// gray pixel index
				unsigned char r = ds_in[ty][tx].x; // red pixel value
				unsigned char g = ds_in[ty][tx].y; // green pixel value
				unsigned char b = ds_in[ty][tx].z; // blue pixel value

				// grayscale conversion using formula 1 from project doc
				d_grey[i] = 0.299f * r + 0.587f * g + 0.114f * b;
				__syncthreads();
			}

			}
			
		/*if (x < numCols && y < numRows) {
			d_grey[i] = ds_grey[threadIdx.y][threadIdx.x];
		}*/
	//}
	//}
	return;
}


void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 


    dim3 block(BLOCK,BLOCK,1);
    dim3 grid((numCols + BLOCK - 1)/BLOCK,(numRows + BLOCK - 1)/BLOCK,1);

    im2Gray_s<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





