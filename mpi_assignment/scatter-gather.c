#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char *argv[])
{
  int numtask, taskid;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numtask);
  MPI_Comm_rank (MPI_COMM_WORLD,&taskid);

  int arr[100], *sendcount, *displs, myarr[100], mycount;
  int i;
  sendcount = malloc (sizeof(int) * numtask);
  displs = malloc (sizeof(int) * numtask);
  if (taskid == 0)
    {
      for (i = 0; i < 100; i++) //For simplicity putting perfect squares of 0-99 in array
        {
          arr[i] = i * i;
        }
      int cur = 100;
      int ideal = cur / numtask; //For completely uniform distribution 
      displs[0] = 0;
      for (i = 0; i < numtask - 1; i++)
        {
          sendcount[i] = ((ideal - 5) + (rand () % 10)); //Ideal uniform distribution +/- 5
          displs[i + 1] = displs[i] + sendcount[i];
          cur -= sendcount[i];
        }
      sendcount[i] = cur;
      for (i = 1; i < numtask; i++)
        {
          MPI_Send (sendcount + i, 1, MPI_INT, i, i, MPI_COMM_WORLD); //Send everyone their count 
        }
      mycount = sendcount[0];
    }
  if (taskid != 0)
    {
      MPI_Recv (&mycount, 1, MPI_INT, 0, taskid, MPI_COMM_WORLD, &status); //Receive my count
    }
  MPI_Scatterv (arr, sendcount, displs, MPI_INT, myarr, mycount, MPI_INT, 0, MPI_COMM_WORLD); //Scatter the array
  for (i = 0; i < mycount; i++)
    {
      myarr[i] = sqrt (myarr[i]);
    }
  MPI_Gatherv (myarr, mycount, MPI_INT, arr, sendcount, displs, MPI_INT, 0, MPI_COMM_WORLD); //Gather at root
  if (taskid == 0) 
    {
      for (i = 0; i < 100; i++)
        {
          printf ("%d ",arr[i]);
        }
      printf ("\n");
    }
  MPI_Finalize ();
  return 0;
}
