#include<stdio.h>
#include<stdlib.h>
#include<time.h>

__device__ int d_change;

__global__ void bellman_ford(int *d_g, int *d_d, int k, int n)
{
        d_change = 0;
        int i = blockIdx.x*blockDim.x+threadIdx.x;
        int cur_dis = d_d[i];
        __syncthreads();
        int j;    
        for (j=1; j<n;j++)
          {
            if (d_g[j*n+i]==1 && cur_dis > d_d[j] + d_g[j*n+i])
              {
                cur_dis = d_d[j] + d_g[j*n+i];
                d_change = 1;
              }
          } 
        __syncthreads();
        d_d[i] = cur_dis;
}

int h_graph[9000][9000];

int main( int argc, char* argv[] )
{
        FILE *fp = fopen("wiki-Vote.txt","r");
        int source =0,dest=0, n =9000,i;
        srand(time(NULL));

        while(!feof(fp))
        {
                fscanf(fp,"%d",&source);
                fscanf(fp,"%d",&dest);
                h_graph[source][dest] = 1;
        }

        fclose(fp);
        int *d_g;

        const size_t a_size = sizeof(int) * size_t(n*n);

        int block_size = atoi(argv[1]);
        int n_blocks = n/block_size + (n%block_size==0?0:1);

        int h_s = 3;
        int h_d[9000], *d_d, k;

        for(i=0; i<n; i++)
                h_d[i] = (int)1e5;

        h_d[h_s] = 0;

        float time;
        cudaEvent_t start, stop;

        cudaEventCreate(&start) ;
        cudaEventCreate(&stop) ;
        cudaEventRecord(start, 0) ;
        cudaMalloc((void **)&d_g, a_size); 
        cudaMemcpy(d_g, h_graph, a_size, cudaMemcpyHostToDevice);
        cudaMalloc(&d_d, n*sizeof(int));
        cudaMemcpy(d_d, h_d,n*sizeof(int),cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime(&time, start, stop) ;

        printf("w %f\n", time);

        cudaEventCreate(&start) ;
        cudaEventCreate(&stop) ;
        cudaEventRecord(start, 0) ;
        for (k=0;k<n-1;k++)
          { 
                bellman_ford<<<n_blocks,block_size>>>(d_g, d_d, k, n);
                int answer;
                cudaMemcpyFromSymbol(&answer, d_change, sizeof(int), 0, cudaMemcpyDeviceToHost);
                if (answer == 0)
                  break;
          }
        cudaEventRecord(stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime(&time, start, stop) ;

        printf("e %f\n", time);

        cudaEventCreate(&start) ;
        cudaEventCreate(&stop) ;
        cudaEventRecord(start, 0) ;
        cudaMemcpy(h_d, d_d,n*sizeof(int),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime(&time, start, stop) ;

        printf("w %f\n", time);
        FILE *op = fopen("bellman.txt","w");
        for (i=0;i<n;i++)
        {
               fprintf(op,"%d: %d\n",i,h_d[i]);
        }
        fclose(op);
        return 0;
}
