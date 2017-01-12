#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned long long int m, first;
#pragma omp threadprivate(first) //making the sequence thread safe

/* \brief Blum Blum Shub Generator
 *
 * Blum Blum Shup is a simple cryptographically secure PRNG
 * R(i+1) = [R(i)*R(i)]/(p1*p2)
 * Where p1 and p2 are large prime numbers
 * Since, the application did not call for an extremely secure PRNG,
 * The length of each prime number are in 10^9, so that the value 
 * p1*p2 lies in range of long long int
 */
double generate ()
{
  first = (first * first) % m;
  return (first * 1.0) / (m * 1.0);
}


/* \brief The main function
 * The function takes 4 command line argument as the two primes, 
 * number of samples and number of threads
 */
int main (int argc, char *argv[])
{
  unsigned long long int p1 = atoll (argv[1]), p2 = atoll (argv[2]);
  int trials = atoi (argv[3]), count = 0;
  int threads = atoi (argv[4]);
  omp_set_num_threads (threads);
  m = p1 * p2;
#pragma omp parallel shared(m) reduction(+:count)
  {
    int *test = malloc (sizeof(int)), i;
    double x, y;
    first = (long long int)time (NULL) * (long long int)test; //Making PRNG numerically correct by giving each thread a seperate sequence
    free (test);
    #pragma omp for
    for (i = 0; i < trials; i++)
      {
        x = generate ();
        y = generate ();

        if ((x * x + y * y) <= 1)
          {
            count++;
          }
      }
  }
  printf ("The value of pie with threads %d = %f\n",threads, (4.0 * count) / trials);
}
