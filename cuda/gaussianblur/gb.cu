#include<stdio.h>
#include<stdlib.h>

__global__ void blur (int *dev_a)
{
 int i = blockIdx.x;
 int j = threadIdx.x;
 int self[3], top[3], bottom[3], left[3], right[3];
 self[0] = dev_a[i*263+j] & 0xff;
 self[1] = (dev_a[i*263+j]>> 8) & 0xff;
 self[2] = (dev_a[i*263+j]>>16) & 0xff;
 if (i==0)
   {
     top[0] = 0;
     top[1] = 0;
     top[2] = 0;
   }
  else
   {
     top[0] = dev_a[(i-1)*263+j]& 0xff;
     top[1] = (dev_a[(i-1)*263+j]>> 8) & 0xff;
     top[2] = (dev_a[(i-1)*263+j]>>16) & 0xff;
   }
  if (i==399)
   {
     bottom[0] = 0;
     bottom[1] = 0;
     bottom[2] = 0;
   }
  else
   {
     bottom[0] = dev_a[(i+1)*263+j]& 0xff;
     bottom[1] = (dev_a[(i+1)*263+j]>> 8) & 0xff;
     bottom[2] = (dev_a[(i+1)*263+j]>>16) & 0xff;
   }
  if (j==0)
   {
     left[0]=0;
     left[1]=0;
     left[2]=0;
   }
  else
   {
     left[0]=dev_a[(i)*263+j-1]& 0xff;
     left[1]=(dev_a[(i)*263+j-1]>> 8) & 0xff;
     left[2]=(dev_a[(i)*263+j-1]>>16) & 0xff;
   }
  if (j==262)
   {
     right[0]=0;
     right[1]=0;
     right[2]=0;
   }
  else
   {
     right[0]=dev_a[(i)*263+j+1]& 0xff;
     right[1]=(dev_a[(i)*263+j+1]>> 8) & 0xff;
     right[2]=(dev_a[(i)*263+j+1]>>16) & 0xff;
   }
  __syncthreads();
  for(int x=0; x<=2;x++)
  self[x] = (top[x]+bottom[x]+left[x]+right[x]+self[x])/5;

  dev_a[i*263+j] = dev_a[i*263+j] & (0xff << 24);
  dev_a[i*263+j] = dev_a[i*263+j] | (self[0]) | (self[1]<<8) | (self[2]<<16);
}

unsigned char header[54];
void ReadBMP(char* filename, int* array)

{
 FILE* img = fopen(filename, "rb");  

fread(header, sizeof(unsigned char), 54, img); 
 int width = *(int*)&header[18];     
 int height = *(int*)&header[22];   
 printf("%d %d\n", width, height);
 int* data = (int *)malloc(width*sizeof(int)); 
 int i;
 for (i=0; i<height; i++ ) {                                     
 fread( data, sizeof(int), width, img);
 int j;
 for (j=0; j<width; j+=1)                                 
 {                       
array[i*width+j] = data[j];                                
}}                       

fclose(img); 

}

void WriteBMP(char* filename, int* array)

{
 FILE* img = fopen(filename, "wb");  
fwrite(header, sizeof(unsigned char), 54, img); 
 int width = *(int*)&header[18];     
 int height = *(int*)&header[22];    
 printf("%d %d\n", width, height);
 int* data = (int *)calloc(width,sizeof(int)); 
 int i;
 for (i=0; i<height; i++ ) {                                     
 int j;
 for (j=0; j<width; j+=1)                                 
 {                       
 data[j] = array[i*width+j];                               

}
 fwrite( data, sizeof(int), width, img);
}                        

fclose(img); 

}

int main()
{
int arr[400*263];
int *dev_a;
unsigned char head[54];
cudaMalloc((void**)&dev_a, 400*263 * sizeof(int));
char name[] = "test.bmp";
ReadBMP(name,arr);
cudaMemcpy(dev_a, arr, 400*263 * sizeof(int), cudaMemcpyHostToDevice);
blur<<<400,263>>>(dev_a);
cudaMemcpy(arr, dev_a, 400*263 * sizeof(int), cudaMemcpyDeviceToHost);
char name1[] = "test1.bmp";
WriteBMP(name1,arr);
}
