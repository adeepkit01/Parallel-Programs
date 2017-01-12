# Programming with OpenMP

All the programs have been written in c, while python has been used for scripting at places instead of bash and also for some complex computations.

### Hello World Program

* Source: hello_world.c
* Compilation: cc hello_world.c -fopenmp -o hello_world
* Execution: ./hello_world

### Hello World Program - Version 2

* Source: hello_worldv2.c
* Compilation: cc hello_world1.c -fopenmp -o hello_worldv2
* Execution: ./hello_worldv2

### DAXPY Loop

* Source: daxpy.c, daxpy_runner.py
* Execution: python daxpy_runner.py
* Output: daxpy_data/
* Remark: The highest sppedup is observed at 2/3 threads. The execution time increases after that which might be because more threads need more switching, hence reducing the efficiency. Sharp increases in the number of threads shows worse performnce than a single thread. Also, increasing the number of threads might lead to different threads working on different pages of memory leading to higher number of swaps.

### Matrix Multiply

* Source: matrix_multiplication.c, matmul_runner.py
* Execution: python matmul_runner.py
* Output: matmul_data/, matmul_dump/
* Remark: Following the trend of DAXPY loop, the highest mean speedup achieved is for 3-5 thread, but in this case increasing the number of threads to 20 is still faster than a serial execution, which might be attributed to the fact that the context switch time in this case will be still less than the serial processing of an O(n^3) algorithm. Each thread calculates every fourth k-th value, where k is the number of threads.

### Calculation of \pi

* Source: pie.c
* Compilation: cc pie.c -fopenmp -o pie
* Execution: ./pie

### Calculation of \pi - Worksharing and Reduction

* Source: pie-for.c
* Compilation: cc pie-for.c -fopenmp -o piefor
* Execution: ./piefor

### Calculation of \pi - Monte Carlo Simulation

* Source: pie-montecarlo-serial.c, pie-montecarlo-parallel.c, monte-carlo_runner.py
* Execution: python monte-carlo_runner.py
* Remark: The reason why we chose Blum Blum Shub pseudo random number generator was because of it having only one value to be seeded for decent randomness, making it easier to be made thread safe, the prime numbers could be easily generated randomly in a single thread making that process thread safe inherently. Also, because of its cryptographic property, it can generate entirely different sequnce by slight change in the seed value, making avoidance of non overlapping sequences easier.

### Producer-Consumer Program

* Source: prod-con.c
* Compilation: cc prod-con.c -fopenmp -o prodcon
* Execution: ./prodcon <Amount of item to be produced>
