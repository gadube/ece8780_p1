#include "im2Gray.h"

#define BLOCK 32

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




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 
    
    dim3 block(1,1,1);
    dim3 grid(numCols,numRows,1);

    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





