/*! \brief Array Helper
 *
 * A helper file to create random real and integral arrays and matrices and to print them
 */ 

#include <time.h>
#include <stdlib.h>

/*! \brief Function to generate random integer array
 *
 * The getArray functions fills the array passed along with its size with random number
 * of upto given range.
 * \param arr Array to be filled
 * \param size Size of the array
 * \param range Range in which randoms numbers are to be generated
 */
void getArray (int arr[], int size, int range)
{
  int i;
  for (i = 0; i < size; i++)
    {
      arr[i] = rand () % range;
    }
}

/*! \brief Function to generate random real array
 *
 * The getdArray functions fills the array passed along with its size with random 
 * real numbers.
 * \param arr Array to be filled
 * \param size Size of the array
 */
void getdArray (double arr[], int size)
{
  int i;
  for (i = 0; i < size; i++)
    {
      arr[i] = (double)rand () / (double)rand ();
    }
}

/*! \brief Function to seed the RNG
 *
 * seed() is used to the seed the random number generator with current system time
 * and should be called by the program using this fill_array utility once
 */
void seed ()
{
  srand (time (NULL));
}

/*! \brief Function to generate random integer matrix
 *
 * The createSqMatrix function allocates memory to the 2-D pointer passed and 
 * fills  with random number of upto given range.
 * \param arr 2-D matrix to be filled
 * \param size Size of row and column of the Matrix
 * \param range Range in which randoms numbers are to be generated
 * \return The final matrix
 */
int** createSqMatrix (int **arr, int size, int range)
{
  int i, j;
  arr = (int **)malloc (size * sizeof(int *));

  for (i = 0; i < size; i++)
    {
      arr[i] = (int *)malloc (size * sizeof(int));
    }
  for (i = 0; i < size; i++)
    {
      for (j = 0; j < size; j++)
        {
          arr[i][j] = rand () % range;
        }
    }

  return arr;
}

/*! brief Function to print the matrix
 *
 * printMatrix funtion prints the passed matrix
 * \param arr 2-D matrix to be printed
 * \param size The size of 2-D matrix to be printed
 */
void printMatrix (int **arr, int size)
{
  int i, j;
  for (i = 0; i < size; i++)
    {
      for (j = 0; j < size; j++)
        {
          printf ("%d ", arr[i][j]);
        }
      printf ("\n");
    }
}
