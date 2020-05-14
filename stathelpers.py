# -*- coding: utf-8 -*-
"""
Small library of helper functions for statistics and probability theory.

Includes classes:
    * Bernoulli
    * Bayes
    * Moivre
    * Poisson
    * RandomVariation
    * StandardDeviation
    * MathExpectation
    * NormalDistribution

@author: DEWHITEE
"""

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
    """
    Bernoulli method is used when we need to calculate series of independent
    experiments.
    
    Note: Is effective if the amount of experiments is less than 20. 
    
    If the count of experiments is higher and probability
    is relatively high (n * p > 9) - use Moivre-Laplace method with the
    Moivre class.
    
    If the count of experiments is higher, but probability is low
    (n * p < 9) - use Poisson method with the Poisson class.
    
    """
    def __init__(self, p, k, n):
        self.out_val = Bernoulli.get(p, k, n)
        self.equation_str = Bernoulli.eqstr(p, k, n)
    
    @staticmethod
    def get(p, k, n):
        return C(k, n) * pow(p, k) * pow((1-p), n-k)
    
    @staticmethod
    def eqstr(p, k, n):
        print("p =",p,"k =",k,"n =",n)
        equationstr = str()
        equationstr += "C("+str(k)+","+str(n)+")"+" * "+str(round(p,4))+"^"+str(k)+" * "
        equationstr += "(1-"+str(round(p,4))+")^("+str(n)+"-"+str(k)+")"
        equationstr += " = " + str(C(k,n))+" * "+str(round(pow(p, k),4))+" * "+str(round(pow((1-p), n-k),4))
        equationstr += " = " + str(round(Bernoulli.get(p,k,n),4))
        return equationstr
    
class Bayes:
    """ 
    Bayes method is used when calculating the probability of an event
    if other event has happened.
    
    To get full (total) probability -- use get_ph(...) method.
    """
    def __init__(self, ah_list, h_list):
        self.out_val = Bayes.get(ah_list, h_list)
        self.full_probability = Bayes.get_ph(ah_list, h_list)
        self.equation_str = Bayes.eqstr(ah_list, h_list)
    
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
    
    @staticmethod
    def eqstr(ah_list, h_list):
        """
        ah_list -- list of probabilities of P(A|H) (can be any list of numerical values)
        h_list -- list of total probabilities of P(H)
        
        returns list of (H|A) probabilities
        """
        print("Full probability =", Bayes.eqstr_ph(ah_list, h_list))
        for j in range(len(h_list)):
            print("P(B"+str(j)+"|A"+str(j)+") = (P(B"+str(j)+")*P(A"+str(j)+"|B"+str(j)+"))/P(A"+str(j)+") = (" + str(h_list[j]) + "*" + str(ah_list[j]) + ") / " + str(Bayes.get_ph(ah_list, h_list))
                  + " = " + str((h_list[j] * ah_list[j]) / Bayes.get_ph(ah_list, h_list)))
    def eqstr_ph(ah_list, h_list):
        """
        ah_list -- list of probabilities of P(A|H) (can be any list of numerical values)
        h_list -- list of total probabilities of P(H)
        
        returns full (total) probability of H
        """
        equationstr = str()
        for j in range(len(h_list)):
            equationstr += "(" + str(h_list[j]) + "*" + str(ah_list[j]) + ")" + " + "
        equationstr = equationstr[:-3] + " = " + str(Bayes.get_ph(ah_list, h_list))
        return equationstr
    
class Moivre:
    """
    Use this method if np > 9 and n > 20
    """
    def __init__(self, p, k, n):
        self.out_val = Moivre.get(p, k, n)
        self.equation_str = Moivre.eqstr(p, k, n)
        #self.equation_str_integral = Mouivre.eqstr_integral(p, k1, k2, n)
        
    @staticmethod
    def x(p, k, n):
        return (k - n * p)/(math.sqrt(n*p*(1-p)))
    @staticmethod
    def get(p, k, n):
        return (1 / (math.sqrt(n*p*(1-p)))) * phi(Moivre.x(p,k,n))
    @staticmethod
    def integral(p,k1,k2,n):
        return norm.cdf(Moivre.x(p,k2,n)) - norm.cdf(Moivre.x(p,k1,n))
    
    @staticmethod
    def eqstr(p, k, n):
        print("p =",p,"k =",k,"n =",n)
        equationstr = str()
        equationstr += "1 / " + "sqrt(" + str(n) + '*' + str(p) + '*(1-' + str(p) + '))'
        equationstr += " * φ(" + str(Moivre.x(p,k,n)) + ") = " + '1 / ' + str(math.sqrt(900*0.8*(1-0.8)))
        equationstr += ' * ' + str(phi(Moivre.x(p, k, n)))
        equationstr += ' = ' + str(Moivre.get(p, k, n))
        return equationstr
    
    @staticmethod
    def eqstr_x(p, k, n):
        equationstr = str()
        equationstr += 'x = (' + str(k) + '-' + str(n) + '*' + str(p)
        equationstr += ' / sqrt(' + str(n) + '*' + str(p) + '* (1-' + str(p) + '))'
        return equationstr
    
    @staticmethod
    def eqstr_integral(p, k1, k2, n):
        equationstr = str()
        print('p =',p,'k1 =',k1,'k2 =',k2,'n =',n)
        print('x1 =',Moivre.eqstr_x(p, k1, n))
        print('x2 =',Moivre.eqstr_x(p, k2, n))
        equationstr += "Ф(x2) - Ф(x1) = " 
        equationstr += "Ф(" + str(Moivre.x(p, k2, n)) + ") - Ф(" + str(Moivre.x(p, k1, n)) + ")" + " = "
        equationstr += str(norm.cdf(Moivre.x(p,k2,n))) + " - " + str(norm.cdf(Moivre.x(p,k1,n)))
        equationstr += " = " + str(Moivre.integral(p,k1,k2,n))
        return equationstr
class Poisson:
    """
    Use this method if np < 9 and n > 20
    """
    def __init__(self, p, k, n):
        self.out_val = Poisson.get(p, k, n)
        self.equation_str = Poisson.eqstr(p, k, n)
    
    @staticmethod
    def get(p, k ,n):
        return ((pow((n*p),k) / math.factorial(k)) * pow(math.e, -n * p))
    
    @staticmethod
    def eqstr(p, k, n):
        equationstr = str()
        print('P =', p, ', k =', k, ', n =', n)
        equationstr += str('(' + str(n) + '*' + str(p) + ')^2 ')
        equationstr += ' / ' + str(k) + '! * ' + str(math.e) + '^(' + str(-n) + ' * ' + str(p) + ')'
        return equationstr + ' = ' + str(Poisson.get(p, k, n))
    
# Random Variation
class RandomVariation:
    """
    """
    def __init__(self, arr):
        self.out_val = RandomVariation.get(arr)
        self.equation_str = RandomVariation.eqstr(arr)
    
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
    
    @staticmethod
    def eqstr(arr):
        """
        arr -- must be an array of pairs, where first argument (key) is xi
        and second (value) is P(X=xi) 
        
        returns the full equation string
        """
        mx = MathExpectation.get(arr)
        equationstr = str()
        localeqstr = str()
        sum = 0
        print("M(X) = " + str(mx) + '; ')
        for key, value in arr.items():
            sum += pow(key-mx, 2) * value
            equationstr += '(' + str(key) + '-' + str(mx) + ")^2 * " + str(value) + ' + '
            localeqstr += str(pow(key-mx, 2) * value) + ' + '
        equationstr = equationstr[:-3]
        localeqstr = localeqstr[:-3]
        equationstr += ' = ' + localeqstr + ' = ' + str(sum)
        return equationstr

class StandardDeviation:
    """
    """
    def __init__(self, arr):
        self.out_val = StandardDeviation.get(arr)
        self.equation_str = StandardDeviation.eqstr(arr)
    
    @staticmethod
    def get(arr):
        return math.sqrt(RandomVariation.get(arr))
    
    @staticmethod
    def eqstr(arr):
        return str("σ(x) = sqrt(D(X)) = sqrt(" + str(RandomVariation.get(arr)) + ") = " + str(math.sqrt(RandomVariation.get(arr))))

class MathExpectation:
    """
    """  
    def __init__(self, arr):
        self.out_val = MathExpectation.get(arr)
        self.equation_str = MathExpectation.eqstr(arr)
    
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
    
    @staticmethod
    def eqstr(arr):
        """
        arr -- must be an array of pairs, where first argument (key) is xi
        and second (value) is P(X=xi) 
        
        returns the full equation string
        """
        sum = 0        
        equationstr = str()
        localeqstr = str()
        for key, value in arr.items():
            sum += key*value
            equationstr += str(key) + ' * ' + str(value) + ' + '
            localeqstr += str(key*value) + ' + '
        equationstr = equationstr[:-3]
        localeqstr = localeqstr[:-3]
        equationstr += ' = ' + localeqstr + ' = ' + str(sum)
        return equationstr
    
class NormalDistribution:
    """
    """
    def __init__(self, x, sigma, mu):
        self.out_val = NormalDistribution.get(x, sigma, mu)
    
    def get(x, sigma, mu):
        return 1 / ((sigma * math.sqrt(2*math.pi)) * pow(math.e, -(pow(x-mu, 2)/(2*pow(sigma, 2)))))
    
        
    