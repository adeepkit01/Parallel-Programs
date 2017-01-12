#include <stdio.h>
#include <omp.h>
#include "fill_arrays.h"

/* A simple producer consumer implementation using openmp*/

int main (int argc, char *argv[])
{
  double *A, sum = 0;
  int flag = 0, N = atoi (argv[1]), i;
  A = (double *)malloc (N * sizeof(double));
  #pragma omp parallel sections shared(A, N, flag, sum) private(i)
  {
    #pragma omp section
    {
      seed ();
      getdArray (A,N);
        #pragma omp flush
      flag = 1;
        #pragma omp flush (flag)
    }
    #pragma omp section
    {
        #pragma omp flush (flag)
      while (flag != 1)
        {
            #pragma omp flush (flag)
        }
          #pragma omp flush
      for (i = 0; i < N; i++)
        {
          sum += A[i];
        }
    }
  }
  printf ("The array = ");
  for (i = 0; i < N; i++)
    {
      printf ("%f ",A[i]);
    }
  printf ("\nThe sum=%f\n", sum);
  return 0;
}
