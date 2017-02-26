import os

os.system ("nvcc -o ssspal sssp-al.cu")

i = 16
while i <= 1024:
  print "# "+str(i)
  os.system("time ./ssspal "+str(i))
  i*=2
