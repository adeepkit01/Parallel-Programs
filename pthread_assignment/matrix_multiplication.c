/*! \brief Matrix Multiply
 *
 * pthread implementation of matrix multiplication.
 */ 

#include <stdio.h>
#include <pthread.h>
#include "fill_arrays.h"
#include <omp.h>
#include <sys/resource.h>

int **x;       //!< The first matrix for matrix multiplication
int **y;       //!< The second matrix for matrix multiplication
int **p;       //!< The product matrix
int arr_size;  //!< The size of the matrix
int thread;    //!< Number of threads to be created

/*! \brief Enum to store modes of execution
 * 
 * The enum helps in the better interaction between the driver script and program
 */
enum mode
{
  output,          //!< To print the operand matrices for multiplication along with product
  time_analysis,   //!< To return the time taken by the procedure
  paging_analysis, //!< To return the stats on page faults 
  check_limits     //!< To return the memory limits on the current program
};

/*! \brief The matrix multiplication loop function
 *
 * The function divides the matrix multiplication loop into the given number of thread. In an 
 * attempt to reduce number of page faults in higher number of threads, every thread works on 
 * every 'thread' row of product starting from their id. For example, if 3 thread are used then 
 * 0th thread will calculate 0, 3, 6, 9... row, 1st thread will calculate 1, 4, 7, 10... row, 
 * and 2nd will calculate 2, 5, 8.... rows of the product matrix
 * \param arg The argument containing the thread ID.
 */
void * matmul (void* arg)
{
  int i, j, k, sum = 0;
  long tid = (long) arg;
  for (i = tid; i < arr_size; i += thread)
    {
      for (j = 0; j < arr_size; j++)
        {
          for (k = 0; k < arr_size; k++)
            {
              sum += (x[i][k] * y[k][j]);
            }
          p[i][j] = sum;
          sum = 0;
        }
    }
  pthread_exit (NULL);
}

/*! \brief The main function
 *
 * The function creates joinable threads based on the request in the command line argument and then joins
 * them after each thread exits. Then the functions returns the set of values based on the mode of execution
 */
int main (int argc, char *argv[])
{
  arr_size = atoi (argv[1]);
  int range = atoi (argv[2]);
  thread = atoi (argv[3]);
  struct rusage usage;
  struct rlimit rla, rld, rlr, rls;
  long i, j;
  seed ();
  x = createSqMatrix (x, arr_size, range);
  y = createSqMatrix (y, arr_size, range);
  p = createSqMatrix (p, arr_size, 1);
  double time_spent = -1 * omp_get_wtime ();
  pthread_t *threads = malloc (sizeof(pthread_t) * thread);
  pthread_attr_t attr;
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
  for (i = 0; i < thread; i++)
    {
      pthread_create (&threads[i], &attr, matmul, (void *) i);
    }
  pthread_attr_destroy (&attr);
  void *status;
  for (i = 0; i < thread; i++)
    {
      pthread_join (threads[i], &status);
    }
  time_spent += omp_get_wtime ();
  enum mode m = atoi (argv[4]);
  if (m == output)
    {
      printf ("The first matrix of product is: \n");
      printMatrix (x, arr_size);
      printf ("The second matrix of product is: \n");
      printMatrix (y, arr_size);
      printf ("The product is: \n");
      printMatrix (p, arr_size);
    }
  else if (m == time_analysis)
    {
      printf ("%f", time_spent);
    }
  else if (m == paging_analysis)
    {
      getrusage (RUSAGE_SELF, &usage);
      printf ("%ld %ld", usage.ru_minflt, usage.ru_maxrss);
    }
  else if (m == check_limits)
    {
      getrlimit (RLIMIT_AS, &rla);
      getrlimit (RLIMIT_DATA, &rld);
      getrlimit (RLIMIT_RSS, &rlr);
      getrlimit (RLIMIT_STACK, &rls);
      printf ("Process virtual memory limit:\t%ld\nData Segment limit:\t%ld\nResident set limit:\t%ld\nStack limit:\t%ld\n", rla.rlim_max, rld.rlim_max, rlr.rlim_max, rls.rlim_max);
    }
  free (x);
  free (y);
  free (p);
  free (threads);
  pthread_exit (NULL);
}

