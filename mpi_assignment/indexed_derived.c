#include <stdio.h>
#include "mpi.h"

#define N  8


int main (int argc, char** argv)
{

  int i, j, k, taskid, numtasks, blocklen[N], disp[N], tag = 42;
  int arr[(N * N)];

  MPI_Datatype upper_triangle;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &taskid);
  MPI_Comm_size (MPI_COMM_WORLD, &numtasks);

  for (i = 0; i < N; i++)
    {
      blocklen[i] = N - i;
      disp[i] = i * N + i;
    }

  MPI_Type_indexed (N, blocklen, disp, MPI_INT, &upper_triangle);
  MPI_Type_commit (&upper_triangle);

  if (taskid == 0)
    {
      for (i = 0; i < (N * N); i++)
        {
          arr[i] = i + 1;
        }

      MPI_Send (arr, 1, upper_triangle, 1, tag, MPI_COMM_WORLD);
    }

  else if (taskid == 1)
    {
      for (i = 0; i < (N * N); i++)
        {
          arr[i] = 0;
        }

      MPI_Recv (arr, 1, upper_triangle, 0, tag, MPI_COMM_WORLD, &status);

      for (i = k = 0; i < N; i++)
        {
          for (j = 0; j < N; j++)
            {
              printf ("%d\t", arr[k++]);
            }
          printf ("\n");
        }
    }

  MPI_Finalize ();
  return 0;
}
