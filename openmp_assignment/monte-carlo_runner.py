import random
import os

"""Generating random numbers to perform fermat test on"""
arr=[]
for i in range(100):
    arr.append(random.randint(10000000, 99999999))

"""Function to calculate (a**n)%m"""
def fast_power(a,n,m):
    result = 1
    value = a%m
    power = n
    while power > 0:
        if power % 2 == 1:
            result = (result * value)%m
        value = (value * value)%m
        power = power/2
    return result%m
        
"""Function which uses Fermat's little theorem to return if number is a probabilistic prime"""
def fermat(x):
    for a in arr:
        if not fast_power(a,x-1,x)==1:
            return False
    return True

n = random.randint(100000000, 999999999)
n1 = random.randint(100000000, 999999999)
a=n%6
n=n+(5-a)
a=n1%6
n1=n1+(5-a)
while True:
    a = fermat(n)
    if a:
        p1=n
        break
    n+=2
    a = fermat(n)
    if a:
        p1=n
        break
    n+=4
while True:
    a = fermat(n1)
    if a:
        p2=n1
        break
    n1+=2
    a = fermat(n1)
    if a:
        p2=n1
        break
    n1+=4

"""p1 and p2 are two large probabilistic primes, the script works fine even during generation of prime numbers in range of 10**50
   but due to limitation of C in the ease of handling such huge number and not a very high requirement of CSPRNG, we continue with
   10**9"""

os.system("cc pie-montecarlo-serial.c -o serial");
os.system("./serial "+str(p1)+" "+str(p2)+" 100000");

os.system("cc pie-montecarlo-parallel.c -fopenmp -o parallel");
for i in range(20):
    os.system("./parallel "+str(p1)+" "+str(p2)+" 100000 "+str(i+1));


os.system("rm serial")
os.system("rm parallel")
