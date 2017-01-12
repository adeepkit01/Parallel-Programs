#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*! \brief Function to print hello world, called by each thread
 *
 * \param tid The argument sent when function is created containing the thread ID
 */
void printHello (int tid)
{
  printf ("Hello World! %d\n", tid);
}

/*! \brief The main function, it creates default number threads and calls PrintHello in each thread
 *
 */
int main ()
{
  int id;
  #pragma omp parallel
  {
    id = omp_get_thread_num ();
    printHello (id);
  }
  return 0;
}
