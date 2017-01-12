"""
Driver script for the matrix multiply loop

The script compiles matrix_multiplication.c and then performs four tasks,
1. It dumps output of exection of one run of number of threds ranging from 1 to 20 in matmul_dump/
2. Then runs the executable with multiple times varying number of threads. The time taken by each 
run is then returned to the script. The script then take mean time for number of threads and plots 
the graph. 
"""

import os
import subprocess

os.system("cc matmul.c -fopenmp -o matmul")

for i in range(20):
    print "Dumping values for thread "+str(i+1)
    os.system("./matmul 1000 100 "+str(i+1)+" 0 > matmul_dump/dump"+str(i+1)+".txt")

maxfile = open('matmul_data/matmul-maxfile.dat', 'w+')
minfile = open('matmul_data/matmul-minfile.dat', 'w+')
avgfile = open('matmul_data/matmul-avgfile.dat', 'w+')

for i in range(20):
    val=[]
    for j in range(100):
        print "Checking time for thread "+str(i+1)+" iteration "+str(j+1)
        a=subprocess.check_output(['./matmul', '250', '100', str(i+1), '1'])
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

os.system("gnuplot -e 'set terminal png size 640,640; set output \"matmul_data/matmul-time.png\"; plot \"matmul_data/matmul-avgfile.dat\" title \"Average time taken per thread\" with linespoint, \"matmul_data/matmul-minfile.dat\" title \"Minimum time taken per thread\" with linespoint, \"matmul_data/matmul-maxfile.dat\" title \"Maximum time taken per thread\" with linespoint'")

os.system("rm matmul")
