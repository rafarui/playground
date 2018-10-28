import sys

def sieve(n):
    """
    Sieve of Eratosthenes, return primes less than n
    """
    mark = [True for i in range(n+1)]
    p=2
    while(p*p <= n ):
        if (mark[p] == True):
            for i in range(2*p,n+1,p):
                mark[i] = False
        p +=1

    primes = []
    for i in range(2,len(mark)):
        if mark[i]:
            primes.append(i)

    return primes

def prime_red(n,primes):
    count_iter = 0
    while (True):
        count_iter += 1
        _sum = 0
        sqrt_n = int(n ** 0.5)
        for i in primes:
            if (i > sqrt_n or n == 1):
                break
            while not n % i: 
                n = n//i
                _sum += i
        if n != 1 and _sum != 0:
            _sum += n
        if _sum == 0:
            break
        n = _sum

    print("%s %s" % (n,count_iter))


primes = sieve(int(10**(9*0.5))+1)
n = int(sys.stdin.readline().strip())
while (n!=4):
    prime_red(n,primes)
    n = int(sys.stdin.readline().strip())
