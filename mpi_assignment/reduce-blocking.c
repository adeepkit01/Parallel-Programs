#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  int numtask, taskid;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numtask);
  MPI_Comm_rank (MPI_COMM_WORLD,&taskid);

  int mynum = taskid;
  while (taskid < numtask) //While I am in the processor list
    {
      if (numtask % 2 != 0 && taskid == 0 && numtask > 2) //If there is an odd process, process 0 receive the odd one
        {
          int add;
          MPI_Recv (&add, 1, MPI_INT, numtask - 1, numtask - 1, MPI_COMM_WORLD, &status);
          mynum += add;
        }

      if (numtask % 2 != 0 && taskid == numtask - 1 && taskid != 0) //If there is an odd process send it to 0 
        {
          MPI_Send (&mynum, 1, MPI_INT, 0, numtask - 1, MPI_COMM_WORLD);
        }

      numtask /= 2; //Put half of processors out of commission 

      if (taskid >= numtask && taskid != 0) //If I am going down, send my value to my peer
        {
          MPI_Send (&mynum, 1, MPI_INT, taskid - numtask, taskid, MPI_COMM_WORLD);
        }
      else if (taskid < numtask) //Receive value from peer
        {
          int add;
          MPI_Recv (&add, 1, MPI_INT, taskid + numtask, taskid + numtask, MPI_COMM_WORLD, &status);
          mynum += add;
        }
    }


  if (taskid == 0)
    {
      printf ("%d\n", mynum);
    }

  MPI_Finalize ();
  return 0;
}
