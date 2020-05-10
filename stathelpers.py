import math
import scipy.integrate as integrate
from scipy.stats import norm

def C(k, n):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def A(k, n):
    return math.factorial(n) / math.factorial(n-k)

def P(n):
    return math.factorial(n)

def phi(x):
    return (1/(math.sqrt(2*math.pi)))*math.exp(-pow(x,2)/2)

def ccc(k, K, n, N):
    return (C(k, K) * C(n - k, N - K)) / C(n, N)
    
class Bernoulli:
    @staticmethod
    def get(p, k, n):
        return C(k, n) * pow(p, k) * pow((1-p), n-k)
    
class Bayes:
    @staticmethod
    def get(ah_list, h_list):
        """
        ah_list -- list of probabilities of P(A|H) (can be any list of numerical values)
        h_list -- list of total probabilities of P(H)
        
        returns list of (H|A) probabilities
        """
        out = []
        for i in range(len(ah_list)):
            sum = 0
            for j in range(len(h_list)):
                sum += h_list[j] * ah_list[j]
            out.append((h_list[i] * ah_list[i]) / sum)
        return out
    
    @staticmethod
    def get_ph(ah_list, h_list):
        """
        ah_list -- list of probabilities of P(A|H) (can be any list of numerical values)
        h_list -- list of total probabilities of P(H)
        
        returns full (total) probability of H
        """
        for i in range(len(ah_list)):
            sum = 0
            for j in range(len(h_list)):
                sum += h_list[j] * ah_list[j]
            return sum
    
    #@staticmethod
    #def get_dict(ahh_dict):
    #    out = dict()
    #    for key, value in ahh_dict.items():
    #        #sum += key * value
    #        out.append((key * value) / key * value)
    #    return out
    
class Moivre:
    @staticmethod
    def x(p, k, n):
        return (k - n * p)/(math.sqrt(n*p*(1-p)))
    @staticmethod
    def get(p, k, n):
        return (1 / (math.sqrt(n*p*(1-p)))) * phi(Moivre.x(p,k,n))
    @staticmethod
    def integral(p,k1,k2,n):
        return norm.cdf(Moivre.x(p,k2,n)) - norm.cdf(Moivre.x(p,k1,n))
    
class Poisson:
    def get(p, k ,n):
        return ((pow((n*p),k) / math.factorial(k)) * pow(math.e, -n * p))
    
# Random Variation
class RandomVariation:
    @staticmethod
    def get(arr):
        """
        arr -- must be an array of pairs, where first argument (key) is xi
        and second (value) is P(X=xi) 
        """
        sum = 0
        mx = MathExpectation.get(arr)
        for key, value in arr.items():
            sum += pow(key-mx, 2) * value
        return sum
    @staticmethod
    def integral(arr, a, b, func):
        mx = MathExpectation.integral(a, b, func)[0]
        return integrate.quad(lambda x: pow(x-mx, 2)*func(x), a, b)

class StandardDeviation:
    @staticmethod
    def get(arr):
        return math.sqrt(RandomVariation.get(arr))

class MathExpectation:
    @staticmethod
    def get(arr):
        """
        arr -- must be an array of pairs, where first argument (key) is xi
        and second (value) is P(X=xi) 
        """
        sum = 0
        for key, value in arr.items():
            sum += key*value
        return sum
    @staticmethod
    def integral(a, b, func):
        return integrate.quad(lambda x: x * func(x), a, b)
    
class NormalDistribution:
    def get(x, sigma, mu):
        return 1 / ((sigma * math.sqrt(2*math.pi)) * pow(math.e, -(pow(x-mu, 2)/(2*pow(sigma, 2)))))
    
        
    