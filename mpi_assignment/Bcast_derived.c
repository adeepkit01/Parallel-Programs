#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpi.h"

/*Structure for the derived data type*/
typedef struct
{
  char c;
  int i[2];
  float f[4];
} dd;

int main (int argc, char *argv[])
{
  int x;
  MPI_Init (&argc, &argv);
  int taskid, numtasks;
  MPI_Comm_rank (MPI_COMM_WORLD, &taskid);
  MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
  dd s;
  MPI_Datatype istruct;
  const int count = 3;
  MPI_Aint disps[count];
  int blocklens[] = {1,2,4};
  MPI_Datatype types[] = {MPI_CHAR, MPI_INT, MPI_FLOAT};

  MPI_Aint block1_address, block2_address, block3_address;
  MPI_Get_address (&s.c, &block1_address);
  MPI_Get_address (s.i, &block2_address);
  MPI_Get_address (s.f, &block3_address);
  disps[0] = 0;
  disps[1] = block2_address - block1_address;
  disps[2] = block3_address - block1_address;

  MPI_Type_create_struct (count, blocklens, disps, types, &istruct);
  MPI_Type_commit (&istruct); //Commit the new data structure as istruct

  if (taskid == 0)
    {
      s.c = 'a';
      s.i[0] = 1;
      s.i[1] = 2;

      for (x = 0; x < 4; x++)
        {
          s.f[x] = (x + 1) * 1.1;
        }

    }

  MPI_Bcast (&s, 1, istruct, 0, MPI_COMM_WORLD);

  printf ("Process %d printing the details of structure \n", taskid);
  printf ("value of char = %c\n", s.c);
  printf ("value of int = %d\t%d\n", s.i[0],s.i[1]);
  printf ("value of float = %f\t%f\t%f\t%f\n", s.f[0],s.f[1],s.f[2],s.f[3]);

  MPI_Finalize ();
  return 0;

}

