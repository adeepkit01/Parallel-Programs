# include <stdlib.h>
# include <cuda.h>
#include<stdio.h>
const int N = 1024; 
__global__ void f(long long int *dev_a) {
    unsigned int tid = threadIdx.x;
        long long int temp = dev_a[(tid+1)%N];
        __syncthreads();
        dev_a[tid] = temp;
}
int main(void) {
    long long int host_a[N];
    long long int *dev_a;
    cudaMalloc((void**)&dev_a, N * sizeof(long long int));
    for(int i = 0 ; i < N ; i++) {
        host_a[i] = i;
    }
    cudaMemcpy(dev_a, host_a, N * sizeof(long long int), cudaMemcpyHostToDevice);
    f<<<1, N>>>(dev_a);
    cudaMemcpy(host_a, dev_a, N * sizeof(long long int), cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < N ; i++) {
        printf("%d ", host_a[i]);
    }
   printf("\n");
}
