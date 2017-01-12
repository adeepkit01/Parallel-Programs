"""
Driver script for the DAXPY loop

The script compiles daxpy.c, then runs the executable with multiple times varying number of threads.
The time taken by each run is then returned to the script. The script then take mean time for number 
of threads and plots the graph. The data and graphs are stored in daxpy_data/.
"""
import os
import subprocess
import random

os.system("cc daxpy.c -pthread -fopenmp -o daxpy")
const = random.randint(0,9)

maxfile = open('daxpy_data/daxpy-maxfile.dat', 'w+')
minfile = open('daxpy_data/daxpy-minfile.dat', 'w+')
avgfile = open('daxpy_data/daxpy-avgfile.dat', 'w+')

for i in range(20):
    val=[]
    for j in range(100):
        a=subprocess.check_output(['./daxpy', str(const), str(i+1)])
        val.append(float(a))
    maxval = max(val)
    minval = min(val)
    avgval = sum(val)/100
    maxfile.write(str(i+1)+' '+str(maxval)+'\n')
    minfile.write(str(i+1)+' '+str(minval)+'\n')
    avgfile.write(str(i+1)+' '+str(avgval)+'\n')

maxfile.close()
minfile.close()
avgfile.close()

os.system("gnuplot -e 'set terminal png size 640,640; set output \"daxpy_data/daxpy.png\"; plot \"daxpy_data/daxpy-avgfile.dat\" title \"Average time taken per thread\" with linespoint, \"daxpy_data/daxpy-minfile.dat\" title \"Min time taken per thread\" with linespoint, \"daxpy_data/daxpy-maxfile.dat\" title \"Max time taken per thread\" with linespoint'")
os.system("rm daxpy")
