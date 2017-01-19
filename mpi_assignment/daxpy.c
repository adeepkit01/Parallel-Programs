#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "fill_arrays.h"
#define ARR_SIZE 65536

double x[ARR_SIZE];
double y[ARR_SIZE];
int a;

int main (int argc, char *argv[])
{
  int numtasks, taskid, dest, offset, i, source, chunksize, rem, acs;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank (MPI_COMM_WORLD,&taskid);
  chunksize = (ARR_SIZE / numtasks); //Size for all processes
  rem = ARR_SIZE % numtasks; //Remainder, which are left after uniform distribution


  if (taskid == 0)
    {
      seed ();
      getdArray (x, ARR_SIZE);
      getdArray (y, ARR_SIZE);
      a = rand()%100;
      double time = -1 * MPI_Wtime ();
      offset = chunksize;
      for (dest = 1; dest < numtasks; dest++)
        {
          if (rem) //Till there is a remainder add one more element for all processes
            {
              acs = chunksize + 1;
              rem--;
            }
          else
            {
              acs = chunksize;
            }
          /*Send elements and constants*/
          MPI_Send (&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
          MPI_Send (&acs, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
          MPI_Send (&x[offset], acs, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
          MPI_Send (&y[offset], acs, MPI_DOUBLE, dest, 4, MPI_COMM_WORLD);
          MPI_Send (&a, 1, MPI_DOUBLE, dest, 5, MPI_COMM_WORLD);
          offset = offset + acs;
        }

      offset = 0;
      for (i = offset; i < (offset + chunksize); i++)
        {
          x[i] = a * x[i] + y[i];
        }

      for (i = 1; i < numtasks; i++)
        {
          source = i;
          /*Receive final answers from all processes*/
          MPI_Recv (&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
          MPI_Recv (&acs, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
          MPI_Recv (&x[offset], acs, MPI_DOUBLE, source, 3, MPI_COMM_WORLD, &status);
          offset += acs;
        }
      time += MPI_Wtime ();
      printf ("%f", time);
    }

  if (taskid > 0)
    {
      source = 0;
      /*Receive all the elements*/
      MPI_Recv (&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
      MPI_Recv (&acs, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv (&x[offset], acs, MPI_DOUBLE, source, 3, MPI_COMM_WORLD, &status);
      MPI_Recv (&y[offset], acs, MPI_DOUBLE, source, 4, MPI_COMM_WORLD, &status);
      MPI_Recv (&a, 1, MPI_DOUBLE, source, 5, MPI_COMM_WORLD, &status);

      for (i = offset; i < (acs + offset); i++)
        {
          x[i] = a * x[i] + y[i];
        }

      dest = 0;
      /*Send the final answer*/
      MPI_Send (&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send (&acs, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
      MPI_Send (&x[offset], acs, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
    }


  MPI_Finalize ();

}




