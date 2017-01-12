/*! \brief DAXPY Loop
 *
 * D stands for Double precision, A is a scalar value, X and Y are one-dimensional 
 * vectors of size 216 each, P stands for Plus. The operation to be completed in one 
 * iteration is X[i] = a*X[i] + Y[i]. The file runs divides the DAXPY loop into specified
 * number of threads and return the time taken in the execution of the procedure. 
 */ 

#include <stdio.h>
#include <pthread.h>
#include "fill_arrays.h"
#include <omp.h>
#define ARR_SIZE 65536

double x[ARR_SIZE]; //!< The first array of DAXPY loop
double y[ARR_SIZE]; //!< The second array of DAXPY loop
int a;              //!< The constant of DAXPY loop
int thread;         //!< Number of threads to be created

/*! \brief The daxpy loop function
 *
 * The function divides the DAXPY loop into the given number of thread. In an attempt to reduce
 * number of page faults in higher number of threads, every thread works on every 'thread' iteration
 * of loop starting from their id. For example, if 3 thread are used then 0th thread will calculate 0,
 * 3, 6, 9..., 1st thread will calculate 1, 4, 7, 10..., and 2nd will calculate 2, 5, 8....
 * \param arg The argument containing the thread ID.
 */
void * daxpy (void* arg)
{
  long i;
  long tid = (long) arg;
  for (i = tid; i < ARR_SIZE; i+=thread)
    {
      x[i] = a * x[i] + y[i];
    }
  pthread_exit (NULL);
}

/*! \brief The main function
 *
 * The function creates joinable threads based on the request in the command line argument and then joins
 * them after each thread exits.
 */
int main (int argc, char *argv[])
{
  a = atoi (argv[1]);
  thread = atoi (argv[2]);
  long i;
  seed ();
  getdArray (x,ARR_SIZE);
  getdArray (y,ARR_SIZE);
  double time_spent = -1 * omp_get_wtime ();
  pthread_t *threads = malloc (sizeof(pthread_t) * thread);
  pthread_attr_t attr;
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
  for (i = 0; i < thread; i++)
    {
      pthread_create (&threads[i], &attr, daxpy, (void *) i);
    }
  pthread_attr_destroy (&attr);
  void *status;
  for (i = 0; i < thread; i++)
    {
      pthread_join (threads[i], &status);
    }
  time_spent += omp_get_wtime ();
  printf ("%f", time_spent);
  free(threads);
  pthread_exit (NULL);
}

