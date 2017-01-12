/*! \brief DAXPY Loop
 *
 * D stands for Double precision, A is a scalar value, X and Y are one-dimensional
 * vectors of size 216 each, P stands for Plus. The operation to be completed in one
 * iteration is X[i] = a*X[i] + Y[i]. The file runs divides the DAXPY loop into specified
 * number of threads and return the time taken in the execution of the procedure.
 */

#include <stdio.h>
#include "fill_arrays.h"
#include <omp.h>
#define ARR_SIZE 65536

double x[ARR_SIZE]; //!< The first array of DAXPY loop
double y[ARR_SIZE]; //!< The second array of DAXPY loop
int a;              //!< The constant of DAXPY loop
int thread;         //!< Number of threads to be created

/*! \brief The main function
 */
int main (int argc, char *argv[])
{
  a = atoi (argv[1]);
  thread = atoi (argv[2]);
  long i;
  int tid;
  seed ();
  getdArray (x,ARR_SIZE);
  getdArray (y,ARR_SIZE);
  double time_spent = -1 * omp_get_wtime ();

  omp_set_dynamic (0);
  omp_set_num_threads (thread);

  #pragma omp parallel shared(a,x,y) private(i,tid)
  {
    /*! \brief The daxpy loop
     */

    #pragma omp for
    for (i = 0; i < ARR_SIZE; i++)
      {
        x[i] = a * x[i] + y[i];

      }
  }
  time_spent += omp_get_wtime ();
  printf ("%f", time_spent);
  return 0;
}

