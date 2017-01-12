#include <stdio.h>
#include <omp.h>

void main ()
{
  double pi, sum = 0.0;
  int num_steps = 1000000000;
  double step = 1.0 / (double)num_steps;

#pragma omp parallel shared(step, sum, num_steps)
  {
    int i;
    int id = omp_get_thread_num ();
    int nthread = omp_get_num_threads ();
    double x, partial;
    for (i = id; i < num_steps; i += nthread)
      {
        x = (i + 0.5) * step;
        partial = partial + 4.0 / (1.0 + x * x);
      }
    #pragma omp atomic
    sum += partial;
  }
  pi = step * sum;
  printf ("%.10lf\n", pi);
}
