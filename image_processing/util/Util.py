import math

def distance(A, B):
    def diff_pow(t):
        a,b = t
        return  (a-b) ** 2
    return math.sqrt(sum(map(diff_pow, zip(A, B))))


