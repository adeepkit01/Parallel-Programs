/*! \brief Hello World Program - 1
 *
 * The file creates 5 threads with the pthread create() routine and passes a thread ID.
 * Each thread prints a “Hello World!” message along with the thread ID, and then 
 * terminates with a call to pthread exit().
 *
 */ 

#include <pthread.h>
#include <stdio.h>

/*! \brief Function to print hello world, called by each thread
 *
 * \param arg The argument sent when function is created containing the thread ID
 */
void * PrintHello (void *arg)
{
  long tid = (long) arg;
  printf ("Hello World! %ld\n", tid);
  pthread_exit (NULL);
}

/*! \brief The main function, it creates 5 threads and calls PrintHello in each thread
 *
 */
int main ()
{
  pthread_t threads[5]; //!< Array of 5 pthreads
  long t;               //!< Thread ID passed to PrintHello
  for (t = 0; t < 5; t++)
    {
      pthread_create (&threads[t], NULL, PrintHello, (void*) t);
    }
  pthread_exit(NULL);
}
