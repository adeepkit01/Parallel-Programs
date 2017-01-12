#include <stdio.h>
#include <omp.h>

void main ()
{
  double pi, sum = 0.0;
  int num_steps = 1000000000;
  double step = 1.0 / (double)num_steps;

#pragma omp parallel shared(step, num_steps) reduction(+:sum)
  {
    int i;
    double x;
    #pragma omp for
    for (i = 0; i < num_steps; i++)
      {
        x = (i + 0.5) * step;
        sum += (4.0 / (1.0 + x * x));
      }
  }
  pi = step * sum;
  printf ("%.10lf\n", pi);
}
