#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  const int tag = 42; /* Message tag */
  int id, ntasks, source_id, dest_id, err, i;
  MPI_Status status;
  char msg[15]; /* Message array */
  sprintf (msg, "Hello World!");
  err = MPI_Init (&argc, &argv); /* Initialize MPI */

  if (err != MPI_SUCCESS)
    {
      printf ("MPI initialization failed!\n");
      exit (1);
    }

  err = MPI_Comm_size (MPI_COMM_WORLD, &ntasks); /* Get nr of tasks */
  err = MPI_Comm_rank (MPI_COMM_WORLD, &id); /* Get id of this process */

  if (id == 0) /* Process 0 (Master Process) does this */
    {
      for (i = 1; i < ntasks; i++)
        {
          err = MPI_Recv (msg, 1, MPI_BYTE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status); /* Receive a message */
          source_id = status.MPI_SOURCE; /* Get id of sender */
          printf ("Received message %s from process %d\n", msg, source_id);
        }
    }
  else  /* Processes 1 to P-1 (senders) do this */
    {
      dest_id = 0; /* Destination address */
      err = MPI_Send (msg, 1, MPI_BYTE, dest_id, tag, MPI_COMM_WORLD);
    }

  MPI_Finalize (); // Terminate MPI 

  return 0;
}
