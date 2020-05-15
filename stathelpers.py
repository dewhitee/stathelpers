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

def prob(m, n):
    return m/n

def wes_prob(m, n):
    print("m =",m,",  n =",n)
    equationstr = "P=m/n="
    equationstr += str(m)+"/"+str(n)+" \\approx "+str(round(m/n,4))
    print(equationstr)

def C(k, n):
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n-k)))

def wes_C(k, n):
    print("k =",k,",  n =",n)
    equationstr = "C^"+str(k)+"_"+str(n)+"=n!/(k!*(n-k)!)="
    equationstr += str(n)+"!"+"/("+str(k)+"!*("+str(n)+"-"+str(k)+")!)"
    equationstr += "="+str(math.factorial(n))+"/("+str(math.factorial(k))+"*"+str(math.factorial(n-k))+")="+str(C(k,n))
    print(equationstr)

def A(k, n):
    return int(math.factorial(n) / math.factorial(n-k))

def wes_A(k, n):
    print("k =",k,",  n =",n)
    equationstr = "A^"+str(k)+"_"+str(n)+"=n!/((n-k)!)="
    equationstr += str(n)+"!"+"/(("+str(n)+"-"+str(k)+")!)"
    equationstr += "="+str(math.factorial(n))+"/("+str(math.factorial(n-k))+")="+str(A(k,n))
    print(equationstr)

def P(n):
    return math.factorial(n)

def wes_P(n):
    print("n =",n)
    print("P_"+str(n)+"=n!="+str(n)+"!"+"="+str(P(n)))

def phi(x):
    return (1/(math.sqrt(2*math.pi)))*math.exp(-pow(x,2)/2)

def ccc(k, K, n, N):
    return (C(k, K) * C(n - k, N - K)) / C(n, N)

def wes_ccc(k, K, n, N):
    print("k =",k,",  K =",K,",  n =",n,",  N =",N)
    equationstr = "P=C^"+str(k)+"_"+str(K)+"*C^("+str(n)+"-"+str(k)+")_("+str(N)+"-"+str(K)+") /C^"+str(n)+"_"+str(N)
    equationstr += "="+str(C(k,K))+"*"+str(C(n-k,N-K))+"/"+str(C(n,N))
    print(equationstr+" \\approx "+str(round(ccc(k,K,n,N),4)))
    
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
        self.equation_str = Bernoulli.es(p, k, n)
    
    @staticmethod
    def get(p, k, n):
        return C(k, n) * pow(p, k) * pow((1-p), n-k)
    
    @staticmethod
    def es(p, k, n):
        """
        Equation string
        """
        print("p =",p,"k =",k,"n =",n)
        equationstr = str()
        equationstr += "C("+str(k)+","+str(n)+")"+" * "+str(round(p,4))+"^"+str(k)+" * "
        equationstr += "(1-"+str(round(p,4))+")^("+str(n)+"-"+str(k)+")"
        equationstr += " = " + str(C(k,n))+" * "+str(round(pow(p, k),4))+" * "+str(round(pow((1-p), n-k),4))
        equationstr += " = " + str(round(Bernoulli.get(p,k,n),4))
        return equationstr
    
    @staticmethod
    def wes(p, k, n):
        """
        Word document equation string
        """
        print("p =",p,",  k =",k,",  n =",n)
        equationstr = str("P{k=")+str(k)+"}="
        equationstr += str("C^k_n p^k (1-p)^(n-k)=")
        equationstr += "C^"+str(k)+"_"+str(n)+" * "+str(round(p,4))+"^"+str(k)+" * "
        equationstr += "(1-"+str(round(p,4))+")^("+str(n)+"-"+str(k)+")"
        equationstr += "=" + str(C(k,n))+" * "+str(round(pow(p, k),4))+" * "+str(round(pow((1-p), n-k),4))
        equationstr += " \\approx " + str(round(C(k,n)*round(pow(p, k),4)*round(pow((1-p), n-k),4),4))
        print(equationstr)
    
class Bayes:
    """ 
    Bayes method is used when calculating the probability of an event
    if other event has happened.
    
    To get full (total) probability -- use get_ph(...) method.
    """
    def __init__(self, ah_list, h_list):
        self.out_val = Bayes.get(ah_list, h_list)
        self.full_probability = Bayes.get_ph(ah_list, h_list)
        self.equation_str = Bayes.es(ah_list, h_list)
    
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
    def es(ah_list, h_list):
        """
        ah_list -- list of probabilities of P(A|H) (can be any list of numerical values)
        h_list -- list of total probabilities of P(H)
        
        returns list of (H|A) probabilities
        """
        print("Full probability =", Bayes.es_ph(ah_list, h_list))
        for j in range(len(h_list)):
            print("P(B"+str(j)+"|A"+str(j)+") = (P(B"+str(j)+")*P(A"+str(j)+"|B"+str(j)+"))/P(A"+str(j)+") = (" + str(h_list[j]) + "*" + str(ah_list[j]) + ") / " + str(Bayes.get_ph(ah_list, h_list))
                  + " = " + str((h_list[j] * ah_list[j]) / Bayes.get_ph(ah_list, h_list)))
            
    def wes(ah_list, h_list, aindex = 0):
        """
        Word document equation string
        """
        datastr_ah = str()
        datastr_h = str()
        for i in range(len(ah_list)):
            datastr_ah += "P(A_"+str(aindex)+"|B_"+str(i)+")="+str(ah_list[i])+",  "
        print(datastr_ah)
        for i in range(len(h_list)):
            datastr_h += "P(B_"+str(i)+")="+str(h_list[i])+",  "
        print(datastr_h)
        Bayes.wes_ph(ah_list, h_list) # full probability
        for j in range(len(h_list)):
            print("P(B_"+str(j)+"|A_"+str(aindex)+") = (P(B_"+str(j)+")*P(A_"+str(aindex)+"|B_"+str(j)+"))/P(A_"+str(aindex)+") = ("+str(h_list[j])+"*"+str(ah_list[j])+")/"+str(Bayes.get_ph(ah_list, h_list))
                  +" \\approx "+str((round((h_list[j]*ah_list[j])/Bayes.get_ph(ah_list, h_list),4))))

    def es_ph(ah_list, h_list):
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
    
    def wes_ph(ah_list, h_list, aindex = 0):
        equationstr = str("P(A_"+str(aindex)+") = ")
        for j in range(len(h_list)):
            equationstr += "("+str(h_list[j])+"*"+str(ah_list[j])+")"+"+"
        equationstr = equationstr[:-1] + " = " + str(Bayes.get_ph(ah_list, h_list))
        print(equationstr)
    
class Moivre:
    """
    Use this method if np > 9 and n > 20
    """
    def __init__(self, p, k, n):
        self.out_val = Moivre.get(p, k, n)
        self.equation_str = Moivre.es(p, k, n)
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
    def es(p, k, n):
        print("p =",p,",  k =",k,",  n =",n)
        equationstr = str()
        equationstr += "1 / " + "sqrt(" + str(n) + '*' + str(p) + '*(1-' + str(p) + '))'
        equationstr += " * φ(" + str(Moivre.x(p,k,n)) + ") = " + '1 / ' + str(math.sqrt(900*0.8*(1-0.8)))
        equationstr += ' * ' + str(phi(Moivre.x(p, k, n)))
        equationstr += ' = ' + str(Moivre.get(p, k, n))
        return equationstr
    
    @staticmethod
    def es_x(p, k, n):
        equationstr = str()
        equationstr += 'x = (' + str(k) + '-' + str(n) + '*' + str(p)
        equationstr += ' / sqrt(' + str(n) + '*' + str(p) + '* (1-' + str(p) + '))'
        return equationstr
    
    @staticmethod
    def es_integral(p, k1, k2, n):
        equationstr = str()
        print('p =',p,',  k1 =',k1,',  k2 =',k2,',  n =',n)
        print('x1 =',Moivre.es_x(p, k1, n))
        print('x2 =',Moivre.es_x(p, k2, n))
        equationstr += "Ф(x2) - Ф(x1) = " 
        equationstr += "Ф(" + str(Moivre.x(p, k2, n)) + ") - Ф(" + str(Moivre.x(p, k1, n)) + ")" + " = "
        equationstr += str(norm.cdf(Moivre.x(p,k2,n))) + " - " + str(norm.cdf(Moivre.x(p,k1,n)))
        equationstr += " = " + str(Moivre.integral(p,k1,k2,n))
        return equationstr
    
    @staticmethod
    def wes(p, k, n):
        print("p =",p,",  k =",k,",  n =",n)
        equationstr = "P{k=" + str(k) + "} = 1/\sqrt(np(1-p)) φ(x) = "
        equationstr += "1/"+"\sqrt("+str(n)+'*'+str(p)+'*(1-'+str(p)+'))'
        equationstr += " φ("+str(round(Moivre.x(p,k,n),4))+")="+'1/'+str(round(math.sqrt(900*0.8*(1-0.8)),4))
        equationstr += '*'+str(round(phi(Moivre.x(p, k, n)),4))
        equationstr += ' \\approx ' + str(round(Moivre.get(p, k, n),5))
        print(equationstr)
        
    @staticmethod
    def wes_integral(p, k1, k2, n):
        equationstr = str()
        print('p =',p,',  k_1 =',k1,',  k_2 =',k2,',  n =',n)
        print('x_1 =',Moivre.wes_x(p, k1, n))
        print('x_2 =',Moivre.wes_x(p, k2, n))
        equationstr += "P{"+str(k1)+"<=k<="+str(k2)+"}="
        equationstr += "Ф(x_2)-Ф(x_1)=" 
        equationstr += "Ф("+str(round(Moivre.x(p, k2, n),4))+")-Ф("+str(round(Moivre.x(p, k1, n),4))+")"+" = "
        equationstr += str(round(norm.cdf(Moivre.x(p,k2,n)),4))+"-"+str(round(norm.cdf(Moivre.x(p,k1,n)),4))
        equationstr += " \\approx " + str(round(Moivre.integral(p,k1,k2,n),5))
        print(equationstr)
        
    @staticmethod
    def wes_x(p, k, n):
        equationstr = str()
        equationstr += '('+str(k)+'-'+str(n)+'*'+str(p)
        equationstr += ')/\sqrt('+str(n)+'*'+str(p)+'*(1-'+str(p)+'))'
        equationstr += " = "+str(round(Moivre.x(p,k,n),4))
        return equationstr
    
class Poisson:
    """
    Use this method if np < 9 and n > 20
    """
    def __init__(self, p, k, n):
        self.out_val = Poisson.get(p, k, n)
        self.equation_str = Poisson.es(p, k, n)
    
    @staticmethod
    def get(p, k ,n):
        return ((pow((n*p),k) / math.factorial(k)) * pow(math.e, -n * p))
    
    @staticmethod
    def es(p, k, n):
        equationstr = str()
        print('P =', p, ', k =', k, ', n =', n)
        equationstr += str('(' + str(n) + '*' + str(p) + ')^2 ')
        equationstr += ' / ' + str(k) + '! * ' + str(math.e) + '^(' + str(-n) + ' * ' + str(p) + ')'
        return equationstr + ' = ' + str(Poisson.get(p, k, n))
    
    @staticmethod
    def wes(p, k, n):
        equationstr = str()
        print('p =', p, ',  k =', k, ',  n =', n)
        print("P_n {X=k}=λ^k/k! e^-λ")
        equationstr += "P_"+str(n)+" {X="+str(k)+"}=λ^k/k! e^-λ="
        equationstr += str('('+str(n)+'*'+str(p)+')^2')
        equationstr += '/'+str(k)+'! *'+str(round(math.e,4))+'^('+str(-n)+'*'+str(p)+')'
        equationstr += "="+str(round((pow((n*p),k) / math.factorial(k)),5))+"*"+str(round(pow(round(math.e,4),-n*p),5))
        print(equationstr+' \\approx '+str(round(round((pow((n*p),k) / math.factorial(k)),5) * round(pow(round(math.e,4),-n*p),5),5)))
    
# Random Variation
class RandomVariation:
    """
    """
    def __init__(self, arr):
        self.out_val = RandomVariation.get(arr)
        self.equation_str = RandomVariation.es(arr)
    
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
    def es(arr):
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
    
    @staticmethod
    def wes(arr):
        mx = round(MathExpectation.get(arr),5)
        equationstr = str()
        localeqstr = str()
        sum = 0
        print("M(X) = " + str(mx))
        equationstr += "D(X)=∑_(i=1)^n▒(x_i -M(X))^2 p_i="
        for key, value in arr.items():
            sum += pow(key-mx, 2) * value
            if mx < 0:
                equationstr += '('+str(round(key,5))+'-('+str(mx)+"))^2 *"+str(round(value,5))+'+'
            else:
                equationstr += '('+str(round(key,5))+'-'+str(mx)+")^2 *"+str(round(value,5))+'+'
            localeqstr += str(round(pow(key-mx, 2)*value,5))+'+'
        equationstr = equationstr[:-1]
        localeqstr = localeqstr[:-1]
        equationstr += '='+localeqstr+' \\approx '+str(round(sum,5))
        print(equationstr)

class StandardDeviation:
    """
    """
    def __init__(self, arr):
        self.out_val = StandardDeviation.get(arr)
        self.equation_str = StandardDeviation.es(arr)
    
    @staticmethod
    def get(arr):
        return math.sqrt(RandomVariation.get(arr))
    
    @staticmethod
    def es(arr):
        return str("σ(x) = sqrt(D(X)) = sqrt(" + str(RandomVariation.get(arr)) + ") = " + str(math.sqrt(RandomVariation.get(arr))))
    
    @staticmethod
    def wes(arr):
        print("σ(x)=\sqrt(D(X))=\sqrt("+str(round(RandomVariation.get(arr),4))+")="+str(round(math.sqrt(RandomVariation.get(arr)),4)))

class MathExpectation:
    """
    """  
    def __init__(self, arr):
        self.out_val = MathExpectation.get(arr)
        self.equation_str = MathExpectation.es(arr)
    
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
    def es(arr):
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
    
    @staticmethod
    def wes(arr):
        """
        arr -- must be an array of pairs, where first argument (key) is xi
        and second (value) is P(X=xi) 
        """
        sum = 0
        for index, key in enumerate(arr):
            print("x_"+str(index)+"="+str(key)+"  ,P{X=x_"+str(index)+"}="+str(arr[key]))
        equationstr = "M(X)=∑_(i=1)^n▒x_i p_i="
        localeqstr = str()
        for key, value in arr.items():
            sum += round(key*value,5)
            equationstr += str(round(key,5))+'*'+str(round(value,5))+'+'
            localeqstr += str(round(key*value,5))+'+'
        equationstr = equationstr[:-1]
        localeqstr = localeqstr[:-1]
        equationstr += '='+localeqstr+' \\approx '+str(round(sum,5))
        print(equationstr)
    
class NormalDistribution:
    """
    """
    def __init__(self, x, sigma, mu):
        self.out_val = NormalDistribution.get(x, sigma, mu)
    
    def get(x, sigma, mu):
        return 1 / ((sigma * math.sqrt(2*math.pi)) * pow(math.e, -(pow(x-mu, 2)/(2*pow(sigma, 2)))))
    
    def prob(mu, sigma, delta):
        return norm.cdf(((mu+delta)-mu)/sigma)-norm.cdf(((mu-delta)-mu)/sigma)
    
    def wes_prob(mu, sigma, delta):
        print("P-?,  μ =",mu,",  σ =",sigma,",  δ =",delta)
        equationstr = "P{"+str(mu-delta)+"<x<"+str(mu+delta)+"}="
        equationstr += "Ф((("+str(mu)+"+"+str(delta)+")-"+str(mu)+")/"+str(sigma)+")-"
        equationstr += "Ф((("+str(mu)+"-"+str(delta)+")-"+str(mu)+")/"+str(sigma)+") ="
        equationstr += "Ф(("+str(mu+delta)+"-"+str(mu)+")/"+str(sigma)+")-"
        equationstr += "Ф(("+str(mu-delta)+"-"+str(mu)+")/"+str(sigma)+") ="
        equationstr += "Ф("+str(((mu+delta)-mu)/sigma)+")-"+"Ф("+str(((mu-delta)-mu)/sigma)+") ="
        equationstr += str(round(norm.cdf(((mu+delta)-mu)/sigma),5))+"-"+str(round(norm.cdf(((mu-delta)-mu)/sigma),5))+" \\approx "
        equationstr += str(round(NormalDistribution.prob(mu,sigma,delta),4))
        print(equationstr)
    
    def prob_interval(mu, sigma, k1, k2):
        """
        Use in case you have an interval like P{k1 < x < k2}
        
        Calculated as: Ф( (k2-mu)/sigma ) - Ф( (k1-mu)/sigma )
        """
        return norm.cdf((k2-mu)/sigma)-norm.cdf((k1-mu)/sigma)
    
    def wes_prob_interval(mu, sigma, k1, k2):
        print("P-?,  μ =",mu,",  σ =",sigma,",  x_1 =",k1,",  x_2 =",k2)
        equationstr = "P{"+str(k1)+"<x<"+str(k2)+"}="
        equationstr += "Ф(("+str(k2)+"-"+str(mu)+")/"+str(sigma)+")-"
        equationstr += "Ф(("+str(k1)+"-"+str(mu)+")/"+str(sigma)+") ="
        equationstr += "Ф("+str(round((k2-mu)/sigma,5))+")-"+"Ф("+str(round((k1-mu)/sigma,5))+") ="
        equationstr += str(round(norm.cdf((k2-mu)/sigma),4))+"-"+str(round(norm.cdf(((k1-mu)/sigma)),4))+" \\approx "
        equationstr += str(round(NormalDistribution.prob_interval(mu,sigma,k1,k2),4))
        print(equationstr)
        
    def prob_nmu(sigma, delta):
        """
        Use in case you have an interval as P{mu - delta < x < mu + delta}
        but the mu argument is undefined.
        
        Ccalculated as: Ф( delta/sigma ) - Ф( -delta/sigma )
        """
        return norm.cdf(delta/sigma)-norm.cdf(-delta/sigma)
    
    def wes_prob_nmu(sigma, delta):
        """
        @see NormalDistribution.prob_nmu(...)
        """
        print("P-?,  μ -?,  σ =",sigma,",  δ =",delta)
        eqstr = "P{μ-"+str(delta)+"<x<μ+"+str(delta)+"}="
        eqstr += "Ф((μ+"+str(delta)+"-μ)/"+str(sigma)+")-"
        eqstr += "Ф((μ-"+str(delta)+"-μ)/"+str(sigma)+")="
        eqstr += "Ф("+str(delta)+"/"+str(sigma)+")-"
        eqstr += "Ф("+str(-delta)+"/"+str(sigma)+")="
        eqstr += "Ф("+str(delta/sigma)+")-"
        eqstr += "Ф("+str(-delta/sigma)+")="
        eqstr += str(round(norm.cdf(delta/sigma),4))+"-"+str(round(norm.cdf(-delta/sigma),4))+" \\approx "
        eqstr += str(round(NormalDistribution.prob_nmu(sigma,delta),4))
        print(eqstr)
    
    def prob_ndelta(mu, sigma, k1, k2):
        """
        Use this, or NormalDistribution.prob_interva(...) instead,
        if you have an interval like P{k1 < x < k2}
        """
        return NormalDistribution.prob_interval(mu, sigma, k1, k2)
    
    def prob_lsigma(sigma, delta):
        """
        Use in case you need to calculate probability as P(|X-a| < sigma) = 2Ф(delta/sigma)
        """
        return 2*(norm.cdf(delta/sigma)-delta)
    
    def wes_prob_lsigma(sigma, delta):
        print("P-?,  μ -?,  σ =",sigma,",  δ =",delta)
        eqstr = "P{|X-a|<"+str(delta)+"}=2Ф("+str(delta)+"/"+str(sigma)+")="
        eqstr += "2Ф("+str(round(delta/sigma,5))+")=2*("+str(round(norm.cdf(delta/sigma),4))+"-"+str(delta)+") \\approx "
        eqstr += str(round(NormalDistribution.prob_lsigma(sigma, delta),4))
        print(eqstr)
    
    
    
    
    