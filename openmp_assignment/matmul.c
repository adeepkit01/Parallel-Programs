/*! \brief Matrix Multiply
 *
 * pthread implementation of matrix multiplication.
 */

#include <stdio.h>
#include "fill_arrays.h"
#include <omp.h>

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
  time_analysis   //!< To return the time taken by the procedure
};


/*! \brief The main function
 *
 * The function creates threads based on the request in the command line argument.
 * Then the thread finds the product
 */
int main (int argc, char *argv[])
{
  arr_size = atoi (argv[1]);
  int range = atoi (argv[2]);
  thread = atoi (argv[3]);
  int i, j, k, sum = 0;
  seed ();
  x = createSqMatrix (x, arr_size, range);
  y = createSqMatrix (y, arr_size, range);
  p = createSqMatrix (p, arr_size, 1);
  double time_spent = -1 * omp_get_wtime ();
  omp_set_dynamic (0);
  omp_set_num_threads (thread);


  #pragma omp parallel shared(x,y,p) private(i,j,k,sum)
  {
    #pragma omp for
    for (i = 0; i < arr_size; i++)
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
  return 0;
}

