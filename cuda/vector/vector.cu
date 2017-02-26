#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}
 
int main( int argc, char* argv[] )
{
    int n = 100000;
 
    double *h_a;
    double *h_b;
    double *h_c;
 
    double *d_a;
    double *d_b;
    double *d_c;
 
    size_t bytes = n*sizeof(double);

    srand(time(NULL));
 
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    for( i = 0; i < n; i++ ) {
        h_a[i] = rand ()%100;
        h_b[i] = rand ()%100;
    }

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("w %f\n", time);
 
    int blockSize, gridSize;
    
    blockSize = atoi(argv[1]);
 
    gridSize = (int)ceil((float)n/blockSize);
 
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("e %f\n", time);
 
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("w %f", time);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
