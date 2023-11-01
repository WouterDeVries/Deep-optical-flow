#include <iostream>
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <typeinfo>
#include <malloc.h>

using namespace pybind11;
using namespace std;

void run_array_op_kernel
(float *X, float *Y, float *Disp, float *coef, int pixels, int samples, 
int *r_samples, int r_iter, int r_sample_num, float sigma, int dim_block);


void RANSAC_GPU(
  pybind11::array_t<float> X, pybind11::array_t<float> Y, pybind11::array_t<float> Disp, pybind11::array_t<float> coef, 
  int pixels, int nsamples, pybind11::array_t<int> r_samples, int r_iter, int r_sample_num, float sigma)
{
    float *mX;
    float *mY;
    float *mDisp;
    float *mcoef;
    int *mr_samples;

    // int *dbug;
    // int *dbug_h;
    // dbug_h = (int *)malloc(pixels * nsamples * sizeof(int));
    // cudaError_t error1 = cudaMalloc(&dbug, pixels * nsamples * sizeof(int));
    // error1 = cudaMemcpy(dbug, dbug_h, pixels * nsamples * sizeof(int), cudaMemcpyHostToDevice);

    auto x_buf = X.request();
    auto y_buf = Y.request();
    auto disp_buf = Disp.request();
    auto coef_buf = coef.request();
    auto r_samples_buf = r_samples.request();
    // cudaDeviceReset();
    cudaError_t error = cudaMalloc(&mX, nsamples * sizeof(float));
    error = cudaMalloc(&mY, nsamples * sizeof(float));
    error = cudaMalloc(&mDisp, pixels * nsamples * sizeof(float));
    error = cudaMalloc(&mcoef, pixels * 3 * sizeof(float));
    error = cudaMalloc(&mr_samples, r_samples.shape(0) * sizeof(int));
    // std::cout << r_samples.shape(0) << std::endl;

    float* mX_h = reinterpret_cast<float*>(x_buf.ptr);
    float* mY_h = reinterpret_cast<float*>(y_buf.ptr);
    float* mDisp_h = reinterpret_cast<float*>(disp_buf.ptr);
    float* mcoef_h = reinterpret_cast<float*>(coef_buf.ptr);
    int* mr_samples_h = reinterpret_cast<int*>(r_samples_buf.ptr);

    error = cudaMemcpy(mX,    mX_h,    nsamples * sizeof(float), cudaMemcpyHostToDevice);
    error = cudaMemcpy(mY,    mY_h,    nsamples * sizeof(float), cudaMemcpyHostToDevice);
    error = cudaMemcpy(mDisp, mDisp_h, pixels * nsamples * sizeof(float), cudaMemcpyHostToDevice);
    error = cudaMemcpy(mcoef, mcoef_h, pixels * 3 * sizeof(float), cudaMemcpyHostToDevice);
    error = cudaMemcpy(mr_samples, mr_samples_h, r_samples.shape(0) * sizeof(int), cudaMemcpyHostToDevice);
    // std::cout << "mem alloc done" << std::endl;
    // std::cout << "r_samples: " << r_samples.shape(0) << std::endl;
    // std::cout << "samples: " << nsamples << std::endl;

    run_array_op_kernel(mX, mY, mDisp, mcoef, pixels, nsamples, mr_samples, r_iter, r_sample_num, sigma, 32);
    // std::cout << "run_array_op_kernel done" << std::endl;

    error = cudaMemcpy(mX_h,    mX,    nsamples * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "cudaMemcpyDeviceToHost mX_h, mX done" << std::endl;
    error = cudaMemcpy(mY_h,    mY,    nsamples * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "cudaMemcpyDeviceToHost mY done" << std::endl;
    error = cudaMemcpy(mDisp_h, mDisp, pixels * nsamples * sizeof(float), cudaMemcpyDeviceToHost);
    error = cudaMemcpy(mcoef_h, mcoef, pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    error = cudaMemcpy(mr_samples_h, mr_samples, r_samples.shape(0) * sizeof(int), cudaMemcpyDeviceToHost);

    // error = cudaMemcpy(dbug_h, dbug, pixels * nsamples  * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < pixels * nsamples; ++i)
    // {
    //     std::cout << "dbug_h: " << dbug_h[i] <<std::endl;
    // }

    error = cudaFree(mX);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaFree(mY);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaFree(mDisp);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaFree(mcoef);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    error = cudaFree(mr_samples);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    // error = cudaFree(dbug);
    // if (error != cudaSuccess) {
    //   throw std::runtime_error(cudaGetErrorString(error));
    // }
    // free(mX_h);
    // free(mY_h);
    // free(mDisp_h);
    // free(mcoef_h);
}

PYBIND11_MODULE(array_op, m)
{
  m.def("RANSAC_GPU", RANSAC_GPU);
}
