/*! \brief Dot Product with Mutex
 *
 * The program computes dot product of two vectors and calculate its running sum. An iteration in a sequential
 * implementation looks like Sum = Sum + (X[i] * Y[i]);. It uses mutex variables for the critical section. Arrays 
 * X and Y, and variable Sum are available to all threads through a globally accessible structure. Each thread 
 * works on a different part of the data. The main thread waits for all the threads to complete their computations, 
 * and then it prints the resulting sum.
 */ 

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include "fill_arrays.h"

/*! \brief The globally accessible structure
 * 
 * The structure variable data is accessible to all the threads
 */
struct global_info
{
  int *X;  //!< The first array for the dot product
  int *Y;  //!< The second array for the dot product
  int Sum; //!< The variable to store the dot product
  int Len; //!< The dimension of the vector
} data;    //!< The globally accessible struct variable

pthread_mutex_t lock; //<! The lock associated with the writing of sum variable 
int thread;           //<! Number of thread

/*! \brief The daxpy loop function
 *
 * The function divides the dot product loop into the given number of thread. In an attempt to reduce
 * number of page faults in higher number of threads, every thread works on every 'thread' iteration
 * of loop starting from their id. For example, if 3 thread are used then 0th thread will calculate 0,
 * 3, 6, 9..., 1st thread will calculate 1, 4, 7, 10..., and 2nd will calculate 2, 5, 8.... When the 
 * thread starts write to the Sum, it acquires lock to prevent race conditions. 
 * \param arg The argument containing the thread ID.
 */
void * dotprod (void *arg)
{
  int i;
  int sum = 0;
  long tid = (long) arg;
  for (i = tid; i < data.Len; i+=thread)
    {
      sum += (data.X[i] * data.Y[i]);
    }
  pthread_mutex_lock (&lock);
  data.Sum += sum;
  pthread_mutex_unlock (&lock);

  pthread_exit (NULL);
}

/*! \brief The main function
 *
 * The function creates joinable threads based on the request in the command line argument and then joins
 * them after each thread exits.
 */
int main (int argc, char *argv[])
{
  if(argc != 4)
    {
      printf("Requires array size, range of random numbers and number of threads as argument\n");
      exit(1);  
    }
  int arr_size = atoi (argv[1]), range = atoi (argv[2]);
  thread = atoi (argv[3]);
  long i;
  seed ();
  data.X = malloc (sizeof(int) * arr_size);
  data.Y = malloc (sizeof(int) * arr_size);
  data.Sum = 0;
  data.Len = arr_size;
  getArray (data.X, arr_size, range);
  getArray (data.Y, arr_size, range);
  pthread_t *threads = malloc (sizeof(pthread_t) * thread);
  pthread_attr_t attr;
  pthread_mutex_init (&lock, NULL);
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
  for (i = 0; i < thread; i++)
    {
      pthread_create (&threads[i], &attr, dotprod, (void *) i);
    }
  pthread_attr_destroy (&attr);
  void *status;
  for (i = 0; i < thread; i++)
    {
      pthread_join (threads[i], &status);
    }
  pthread_mutex_destroy (&lock);
  printf ("The dot product is %d\n",data.Sum);
  free(threads);
  free(data.X);
  free(data.Y);
  pthread_exit (NULL);
}
