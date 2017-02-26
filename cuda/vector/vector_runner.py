import os

os.system ("nvcc -o vector vector.cu")

i = 16
while i <= 1024:
  print "# "+str(i)
  os.system("time ./vector "+str(i))
  i*=2
