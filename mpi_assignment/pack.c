#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpi.h"

typedef struct
{
  char c;
  int i[2];
  float f[4];
} dd;

int main (int argc, char *argv[])
{

  int x;
  const int tag = 42;
  MPI_Status status;
  int taskid, numtasks;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &taskid);
  MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
  dd s;
  char buffer[100];
  int position = 0;

  if (taskid == 0)
    {
      s.c = 'a';
      s.i[0] = 1;
      s.i[1] = 2;

      for (x = 0; x < 4; x++)
        {
          s.f[x] = (x + 1) * 1.1;
        }

      MPI_Pack (&s.c, 1, MPI_CHAR, buffer,100,&position,MPI_COMM_WORLD);
      MPI_Pack (s.i, 2, MPI_INT, buffer, 100, &position,MPI_COMM_WORLD);
      MPI_Pack (s.f, 4, MPI_FLOAT,buffer,100, &position,MPI_COMM_WORLD);

      for (x = 1; x < numtasks; x++)
        {
          MPI_Send (buffer, position, MPI_PACKED, x, tag, MPI_COMM_WORLD);
        }
    }
  else
    {

      MPI_Recv (buffer, 100, MPI_PACKED, 0, tag, MPI_COMM_WORLD, &status);
      position = 0;
      MPI_Unpack (buffer, 100, &position, &s.c, 1, MPI_CHAR, MPI_COMM_WORLD);
      MPI_Unpack (buffer, 100, &position, s.i, 2, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack (buffer, 100, &position, s.f, 4, MPI_FLOAT, MPI_COMM_WORLD);

      printf ("Process %d printing the details of structure \n", taskid);
      printf ("value of char = %c\n", s.c);
      printf ("value of int = %d\t%d\n", s.i[0],s.i[1]);
      printf ("value of float = %f\t%f\t%f\t%f\n", s.f[0],s.f[1],s.f[2],s.f[3]);
    }

  MPI_Finalize ();
  return 0;

}

