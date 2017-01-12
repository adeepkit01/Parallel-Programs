/*! \brief Hello World Program - 0
 *
 * The file creates 5 threads with the pthread create() routine. Each thread prints a 
 * “Hello World!” message, and then terminates with a call to pthread exit().
 *
 */ 
#include <pthread.h>
#include <stdio.h>

/*! \brief Function to print hello world, called by each thread
 *
 */
void * PrintHello ()
{
  printf ("Hello World! \n");
  pthread_exit (NULL);
}

/*! \brief The main function, it creates 5 threads and calls PrintHello in each thread
 *
 */
int main ()
{
  pthread_t threads[5];  //!< Array of 5 pthreads
  int t;
  for (t = 0; t < 5; t++)
    {
      pthread_create (&threads[t], NULL, PrintHello, NULL);
    }
  pthread_exit(NULL);
}
