#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  int numtask, taskid;
  MPI_Status *status;
  MPI_Request *request;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numtask);
  MPI_Comm_rank (MPI_COMM_WORLD,&taskid);

  int mynum = taskid;
  int count = 0;
  int *add;
  status = malloc (sizeof(MPI_Status) * numtask);
  request = malloc (sizeof(MPI_Request) * numtask);
  add = malloc (sizeof(int) * numtask); //Each process now has an array where it receives values from all its peer
  while (taskid < numtask / 2) //While I am not going down 
    {
      if (numtask % 2 != 0 && taskid == 0 && numtask > 2)  //Process 0 should still receive the extra odd
        {
          MPI_Irecv (add + count, 1, MPI_INT, numtask - 1, numtask - 1, MPI_COMM_WORLD, request + count);
          count++;
        }
      numtask /= 2;
      if (taskid < numtask) //If I am active, my peer will send me something 
        {
          MPI_Irecv (add + count, 1, MPI_INT, taskid + numtask, taskid + numtask, MPI_COMM_WORLD, request + count);
          count++;
        }
    }
  MPI_Waitall (count, request, status); //Now wait for all recv
  int i;
  for (i = 0; i < count; i++) //Add all my peer numbers
    {
      mynum += add[i];
    }
  if (numtask % 2 != 0 && taskid == numtask - 1 && taskid != 0) //If I am the last odd, 0 is my peer
    {
      MPI_Isend (&mynum, 1, MPI_INT, 0, numtask - 1, MPI_COMM_WORLD, request + count);
      MPI_Wait (request + count, status + count);
    }
  numtask /= 2;
  if (taskid >= numtask && taskid != 0) //Send to my peer
    {
      MPI_Isend (&mynum, 1, MPI_INT, taskid - numtask, taskid, MPI_COMM_WORLD, request + count);
      MPI_Wait (request + count, status + count);
    }
  if (taskid == 0)
    {
      printf ("%d\n", mynum);
    }
  MPI_Finalize ();
  return 0;
}
