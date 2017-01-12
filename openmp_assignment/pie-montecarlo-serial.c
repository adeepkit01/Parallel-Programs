#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned long long int m, first;

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
 * The function takes 3 command line argument as the two primes and 
 * number of samples
 */
int main (int argc, char *argv[])
{
  unsigned long long int p1 = atoll (argv[1]), p2 = atoll (argv[2]); 
  m = p1 * p2;
  int *test = malloc (sizeof(int)), i;
  double x, y;
  first = (long long int)time (NULL) * (long long int)test;
  free (test);
  int trials = atoi (argv[3]), count = 0;
  for (i = 0; i < trials; i++)
    {
      x = generate ();
      y = generate ();

      if ((x * x + y * y) <= 1)
        {
          count++;
        }
    }
  printf ("The value of pie from serial program = %f\n", (4.0 * count) / trials);
}
