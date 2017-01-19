#include <mpi.h>
#include <stdio.h>
#include<stdlib.h>
int main (int argc, char *argv[])
{
  int done = 0, taskid, numtasks, i;
  double ans, pi, h, partial, x, a;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank (MPI_COMM_WORLD, &taskid);
  int num_steps;
  /*Initialise num_steps in 0th process*/
  if(taskid == 0)
  {
    num_steps = 1000000000;
  }

  MPI_Bcast (&num_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  double step = 1.0 / (double)num_steps;
  partial = 0.0;

  for (i = taskid+1 ; i <=num_steps; i += numtasks)
    {
       x = (i - 0.5) * step;
       partial = partial + 4.0 / (1.0 + x * x);
    }
  ans = step * partial;
  MPI_Reduce (&ans, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (taskid == 0)
    {
      printf ("pi is approximately %lf \n", pi);
    }
  MPI_Finalize ();
  return 0;
}
