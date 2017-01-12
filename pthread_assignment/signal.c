/*! \brief Signalling using Condition Variables
 *
 * The program Creates 2 increment-count threads and 1 watch-count thread. Increment-count 
 * threads increment a count variable (shared by both) till a threshold is reached. On 
 * reaching the threshold, a signal is sent to the watch-count thread (using pthread_cond_signal).
 * The watch-count thread locks the count variable, and waits for the signal (using 
 * pthread_cond_wait) from one of the increment-count threads. As signal arrives, the watch
 * -count thread releases lock and exits. The other two threads exit too.
 */ 

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t lock;         //!< The lock associated with the counter
pthread_cond_t cond_variable; //!< The conditional variable for signal
int count;                    //!< The counter variable
int count_limit;              //!< The threshold limit for counter

/*! \brief The function for increment count
 *
 * The function keeps incrementing the value of count will the threshold is reached,
 * Then it signals on cond_variable and calls pthread_exit.
 */
void * increment ()
{
  printf ("Increment thread Starts\n");
  while (1)
    {
      pthread_mutex_lock (&lock);
      count++;
      if (count == count_limit)
        {
          pthread_cond_signal (&cond_variable);
          printf ("Signal sent\n");
        }
      if (count > count_limit)
        {
          pthread_mutex_unlock (&lock);
          break;
        }
      pthread_mutex_unlock (&lock);
    }
  printf ("Increment thread Exits\n");
  pthread_exit (NULL);
}

/*! \brief The function for watch count
 *
 * The function keeps waiting for a signal from increment counter about the counter 
 * reaching the threshold and then exits.
 */
void * watch ()
{
  printf ("Watching thread started\n");
  pthread_mutex_lock (&lock);
  if (count < count_limit)
    {
      printf ("Watching thread waiting\n");
      pthread_cond_wait (&cond_variable, &lock);
    }
  printf ("Value of count in Watch is %d\n", count);
  pthread_mutex_unlock (&lock);
  printf ("Watching thread Exits\n");
  pthread_exit (NULL);
}

/*! \brief The main function
 *
 * The function creates joinable threads based on the request in the command line argument and then joins
 * them after each thread exits.
 */
int main (int argc, char *argv[])
{
  if(argc != 2)
    {
      printf("Requires count threshold as argument\n");
      exit(1);  
    }
  count_limit = atoi (argv[1]);
  count = 0;
  int i;
  pthread_t threads[3];
  pthread_attr_t attr;
  pthread_mutex_init (&lock, NULL);
  pthread_cond_init (&cond_variable, NULL);

  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create (&threads[0], &attr, watch, NULL);
  pthread_create (&threads[1], &attr, increment, NULL);
  pthread_create (&threads[2], &attr, increment, NULL);
  pthread_attr_destroy (&attr);
  void *status;
  for (i = 0; i < 3; i++)
    {
      pthread_join (threads[i], &status);
    }
  pthread_attr_destroy (&attr);
  pthread_mutex_destroy (&lock);
  pthread_cond_destroy (&cond_variable);
  pthread_exit (NULL);
}
