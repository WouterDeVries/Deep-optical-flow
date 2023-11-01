#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand_kernel.h>

using namespace std;

__device__ float getElement(float *A, int row, int col, int height, int width)
{
    return A[row * width + col];
}


__device__ void setElement(float *A, int row, int col, float value, int height, int width)
{
    A[row * width + col] = value;
}


__device__ float gDeterm3(
    float a0, float a1, float a2, 
    float a3, float a4, float a5, 
    float a6, float a7, float a8
)
{
    float sum1, sum2;
    sum1=a2*a3*a7+a1*a5*a6+a0*a4*a8;
    sum2=a8*a1*a3+a7*a5*a0+a2*a4*a6;
    return sum1-sum2;
}


__device__ bool gFittingPlane(float *x, float *y, float *z, int n, float &a, float &b, float &c)
{
    int i;
    float x1, x2, y1, y2, z1, xz, yz, xy, r;
 
    x1 = x2 = y1 = y2 = z1 = xz = yz = xy = 0.0f;
    for(i=0; i<n; ++i)
    {
        x1 += x[i];
        x2 += x[i]*x[i];
        xz += x[i]*z[i];
 
        y1 += y[i];
        y2 += y[i]*y[i];
        yz += y[i]*z[i];
 
        z1 += z[i];
        xy += x[i]*y[i];
    }
 
    r = gDeterm3(x2, xy, x1, xy, y2, y1, x1, y1, n) + 1e-5;
    // if(r == 0) return false;
 
    a = gDeterm3(xz, xy, x1, yz, y2, y1, z1, y1, n) / r;
    b = gDeterm3(x2, xz, x1, xy, yz, y1, x1, z1, n) / r;
    c = gDeterm3(x2, xy, xz, xy, y2, yz, x1, y1, z1) / r;
 
    return true;
}


__device__ void RANSAC_FitPlane(float *X, float *Y, float *Disp, float *coef, int samples, int *r_samples, int r_iter, int r_sample_num, float sigma)
{
    int max_inliers = 0; 
    float best_dx = 0.0;
    float best_dy = 0.0;
    float best_d  = 0.0;
    for (int iter = 0; iter < r_iter; ++iter)
    {
        float dx = 0.0;
        float dy = 0.0;
        float d  = 0.0;
        
        // ----------- At least square regression for a window ------------- //
        float x1, x2, y1, y2, z1, xz, yz, xy, r;
        x1 = x2 = y1 = y2 = z1 = xz = yz = xy = 0.0f;
        for(int i=0; i<r_sample_num; ++i)
        {
            int index = r_samples[i+iter*r_sample_num];
            x1 += X[index];
            x2 += X[index]*X[index];
            xz += X[index]*Disp[index];
     
            y1 += Y[index];
            y2 += Y[index]*Y[index];
            yz += Y[index]*Disp[index];
     
            z1 += Disp[index];
            xy += X[index]*Y[index];

            // dbug[index] = ceil(Y[index]);
        }
        r  = gDeterm3(x2, xy, x1, xy, y2, y1, x1, y1, r_sample_num) + 1e-15;
        dx = gDeterm3(xz, xy, x1, yz, y2, y1, z1, y1, r_sample_num) / r;
        dy = gDeterm3(x2, xz, x1, xy, yz, y1, x1, z1, r_sample_num) / r;
        d  = gDeterm3(x2, xy, xz, xy, y2, yz, x1, y1, z1) / r;
        // *********** At least square regression for a window *********** //


        int inliers = 0;
        for (int i = 0; i < samples; ++i)  // get inliers number of a window
        {
            float err = dx * X[i] + dy * Y[i] + d - Disp[i];  //cal regression error of a pixel
            float abs_err;

            if (err < 0)
                abs_err = 0 - err;
            else
                abs_err = err;  // abs(err)

            if (abs_err < sigma)
                inliers++;
        }
        if (inliers >= max_inliers)  // update the best plane params
        {
            max_inliers = inliers;
            best_dx = dx;
            best_dy = dy;
            best_d = d;
        }
    }
    coef[0] = best_dx;
    coef[1] = best_dy;
    coef[2] = best_d;
}

__global__ void triMatOp(float *X, float *Y, float *Disp, float *coef, int pixels, int samples, 
    int *r_samples, int r_iter, int r_sample_num, float sigma)
{
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //  throw std::runtime_error(cudaGetErrorString(error));
    // }
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = row; r < pixels; r += stride)
    {
        // dbug[samples*r] = r;
        RANSAC_FitPlane(X, Y, &Disp[samples*r], &coef[3*r], samples, r_samples, r_iter, r_sample_num, sigma);
    }
}


void run_array_op_kernel
(float *X, float *Y, float *Disp, float *coef, int pixels, int samples, 
    int *r_samples, int r_iter, int r_sample_num, 
    float sigma, int dim_block)
{
//   cudaError_t error = cudaGetLastError();
//   if (error != cudaSuccess) {
//       std::stringstream strstr;
//       strstr << cudaGetErrorString(error);
//       throw strstr.str();
//     }
  dim3 dimBlock(dim_block, 1, 1);
//   std::cout << "dimBlock.x: " << dimBlock.x << std::endl;
  dim3 dimGrid(ceil((float)pixels / dimBlock.x));
  triMatOp<<<dimGrid, dimBlock>>>(X, Y, Disp, coef, pixels, samples, r_samples, r_iter, r_sample_num, sigma);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_array_op_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
