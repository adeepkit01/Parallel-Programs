import os

os.system ("nvcc -o sssp sssp.cu")

i = 16
while i <= 1024:
  print "# "+str(i)
  os.system("time ./sssp "+str(i))
  i*=2
