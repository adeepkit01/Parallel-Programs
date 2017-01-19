# Programming with MPI

All the programs have been written in c, while python has been used for scripting at places instead of bash and also for some complex computations.

### Hello World Program

* Source: HelloWorld.c
* Compilation: mpicc -o HelloWorld ./HelloWorld.c
* Execution: mpirun -n <number of processes> ./HelloWorld

### DAXPY Loop

* Source: daxpy.c, daxpy_runner.py
* Execution: python daxpy_runner.py
* Output: daxpy_data/
* Remark: The MPI execution continuously shows a slowing down of execution with increasing number of processes, which can be attributed to DAXPY being an O(n) algorithm with simple computing complexity, does not gain with the high communication cost of MPI.

### Hello World Program - Version 2

* Source: HelloWorldv2.c
* Compilation: mpicc -o HelloWorldv2 ./HelloWorldv2.c
* Execution: mpirun -n <number of processes> ./HelloWorldv2

### Calculation of \pi

* Source: pie.c
* Compilation: mpicc -o pie ./pie.c
* Execution: mpirun -n <number of processes> ./pie

### Reduction operation

* Source: reduce-blocking.c reduce-nonblocking.c
* Compilation: mpicc -o <binary name> ./<program name>
* Execution: mpirun -n 4 ./<binary name>

### Collective Communication - Scatter - Gather

* Source: scatter-gather.c
* Compilation: mpicc -o scatter-gather ./scatter-gather.c -lm
* Execution: mpirun -n <number of processes> ./scatter-gather

### MPI Derived Datatypes

* Source: Bcast_derived.c Send_Recv_derived.c
* Compilation: mpicc -o <binary name> ./<program name>
* Execution: mpirun -n 4 ./<binary name>

### Pack and Unpack

* Source: pack.c
* Compilation: mpicc -o pack ./pack.c
* Execution: mpirun -n <number of processes> ./pack

### Derived Datatype - Indexed

* Source: indexed_derived.c
* Compilation: mpicc -o indexed_derived ./indexed_derived.c
* Execution: mpirun -n <number of processes> ./indexed_derived

### Matrix Multiplication on a Cartesian Grid (2D Mesh) using Cannon’s Algorithm

* Source: matmul.c
* Compilation: mpicc -o matmul ./matmul.c
* Execution: mpirun -n 4 ./matmul
