# Programming with POSIX Threads

All the programs have been written in c, while python has been used for scripting at places instead of bash.

### Hello World Program - 0

* Source: hello_world0.c
* Compilation: cc hello_world0.c -pthread -o hello_world0
* Execution: ./hello_world0

### Hello World Program - 1

* Source: hello_world1.c
* Compilation: cc hello_world1.c -pthread -o hello_world1
* Execution: ./hello_world1

###  Know Your System

* Final Document: know_your_system.pdf
* Information sources: know your system supporting file/ and different benchmarking tools

### DAXPY Loop

* Source: daxpy.c, daxpy_runner.py
* Execution: python daxpy_runner.py
* Output: daxpy_data/
* Remark: The highest sppedup is observed at 3/4 threads. The execution time increases after that which might be because more threads need more switching, hence reducing the efficiency. Sharp increases in the number of threads shows worse performnce than a single thread. Also, increasing the number of threads might lead to different threads working on different pages of memory leading to higher number of swaps.

###  Dot Product with Mutex

* Source: dot_product.c
* Compilation: cc dot_product.c -pthread -o dot_product
* Execution: ./dot_product <size of array> <range of random number> <number of threads>

### Signalling using Condition Variables

* Source: signal.c
* Compilation: cc signal.c -pthread -o signal
* Execution: ./signal <count threshold>

### Matrix Multiply

* Source: matrix_multiplication.c, matmul_runner.py
* Execution: python matmul_runner.py
* Output: matmul_data/, matmul_dump/
* Remark: Following the trend of DAXPY loop, the highest mean speedup achieved is for 3-5 thread, but in this case increasing the number of threads to 20 is still faster than a serial execution, which might be attributed to the fact that the context switch time in this case will be still less than the serial processing of an O(n^3) algorithm. The largest value for which the program has been tested is a 3700*3700 matrix which gave 40375 page faults. Since the limit found is unlimited (-1), and since the matmul code uses heap for storing arrays, the program can theoritically run larger matrices, but running for 3700*3700 matrix on 4 thread takes well over 3 minutes, hence larger array will go on taking more time.
