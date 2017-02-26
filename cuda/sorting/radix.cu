#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ void partition_by_bit(unsigned int *values, unsigned int bit);


__global__ void radix_sort(unsigned int *values)
{
    int  bit;
    for( bit = 0; bit < 32; ++bit )
    {
        partition_by_bit(values, bit);
        __syncthreads();
    }
}

__device__ int plus_scan(unsigned int *x)
{
    unsigned int i = threadIdx.x; // id of thread executing this instance
    unsigned int n = blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        unsigned int t;

        if ( i >= offset ) 
            t = x[i-offset];
        
        __syncthreads();

        if ( i >= offset ) 
            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]

        __syncthreads();
    }
    return x[i];
}
__device__ void partition_by_bit(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];          // value of integer at position i
    unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit
    values[i] = p_i;  
    __syncthreads();
    unsigned int T_before = plus_scan(values);
    unsigned int T_total  = values[size-1];
    unsigned int F_total  = size - T_total;
    __syncthreads();
    if ( p_i )
        values[T_before-1 + F_total] = x_i;
    else
        values[i - T_before] = x_i;
}

int main ()
{
unsigned int a[1000];
int size = 1000;
srand(time(NULL));
for (int i = 0; i < 1000; i++)
{
  a[i] = rand ()%1000;
}
unsigned int *dev_a;
cudaMalloc(&dev_a, size * sizeof(unsigned int));
cudaMemcpy( dev_a, a, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
radix_sort<<<1,size>>>(dev_a);
cudaMemcpy( a, dev_a, size * sizeof(unsigned int), cudaMemcpyDeviceToHost );
for (int i = 0; i < 1000; i++)
{
  printf("%u ", a[i]);
}
printf ("\n");
} 

