"""
Copyright: Copyright (C) 2015 Baruch College - Interest Rate Model
Description: B-splines library
Author: Poyuan Huang
"""

# Python imports

from numpy import *
from scipy.interpolate import splrep, splev, splint
from matplotlib.pyplot import plot, show

def test_splrep():

    x = linspace(0,10,10)
    y = sin(x)
    tck = splrep(x, y)
    #print (tck)


def test_splev():

    x = linspace(0,10,10)
    y = cos(x)
    tck = splrep(x, y)
    x2 = linspace(0,10,10)
    y2 = splev(x2, tck)
    plot(x,y,'o',x2,y2)
    show()


def test_splint():
 
    x = linspace(0,10,10)
    y = sin(x)
    tck = splrep(x, y)
    y2 = splint(x[0],x[-1],tck)
    #print (y2)
        


class Spline(object):
    ''' B-spline class '''
    def __init__(self, ls_knots):
        '''
        @summary: B-spline constructor
        @param t: type of list, the vector of knots
        '''
        super(Spline, self).__init__()
        self.ls_knots       = ls_knots
        self.d_cache        = {}
        self.d_cache_gamma  = {}
        self.d_cache_crsint = {}


    def splrep(self, i_start, i_degree, f_time):
        '''
        @summary: B-spline functions
        @param i_start: start index
        @param i_degree: B-spline degree
        @param f_time: time 
        '''
        f_begin = self.ls_knots[i_start]
        f_end   = self.ls_knots[i_start+i_degree+1]
        if f_time < f_begin or f_time >= f_end:
            return 0.
        elif i_degree == 0:
            return 1.
        else:
            if (i_start, i_degree, f_time) in self.d_cache.keys():
                return self.d_cache[(i_start, i_degree, f_time)]
            else:
                if (i_start, i_degree, f_time) in self.d_cache.keys():
                    return self.d_cache[(i_start, i_degree, f_time)]
                else:
                    self.d_cache[(i_start ,i_degree ,f_time)] = (f_time-f_begin) / (self.ls_knots[i_start+i_degree]-f_begin) * self.splrep(i_start,   i_degree-1, f_time) \
                                                              + (f_end -f_time)  / (f_end-self.ls_knots[i_start+1])          * self.splrep(i_start+1, i_degree-1, f_time)
                    return self.d_cache[(i_start ,i_degree ,f_time)]


    def splint(self, i_start, i_degree, f_time):
        '''
        @summary: B-spline integration
        @param i_start: start index
        @param i_degree: B-spline degree
        @param f_time: time 
        '''
        f_begin = self.ls_knots[i_start]
        f_end   = self.ls_knots[i_start+i_degree+1]
        if f_time < f_begin:
            return 0.
        elif f_time >= f_end:
            return (f_end-f_begin) / (i_degree+1)
        else:
            f_sum = 0.
            while self.ls_knots[i_start] < f_time:
                f_sum += (f_end - f_begin) / (i_degree + 1) * self.splrep(i_start, i_degree + 1, f_time)
                i_start += 1
            return f_sum


    def splder(self, i_start, i_degree, f_time, order):
        '''
        summary: B-spline derivative
        @param i_start: start index
        @param i_degree: B-spline degree
        @param f_time: time 
        @param i_order: highest order
        '''
        f_begin = self.ls_knots[i_start]
        f_end   = self.ls_knots[i_start+i_degree+1]
        if order == 0:
            return self.splrep(i_start, i_degree, f_time)
        elif order == 1 and i_degree < 1.:
                return 0.
        else:
            return i_degree / ( self.ls_knots[i_start+i_degree]-f_begin) * self.splder(i_start,   i_degree-1, f_time, order-1) \
                 + i_degree / (-self.ls_knots[i_start+1]       +f_end  ) * self.splder(i_start+1, i_degree-1, f_time, order-1)


    def splgamma(self, i_start, f_start, f_end):
        ''' B-spline gamma '''
        if (i_start, f_start, f_end) in self.d_cache_gamma.keys():
            pass
        else:
            self.d_cache_gamma[(i_start, f_start, f_end)] = self.splint(i_start,3,f_end) - self.splint(i_start,3,f_start)
        return self.d_cache_gamma[(i_start, f_start, f_end)]


    def splcrsint(self, i_start, i_start2, f_start, f_end):
        ''' B-spline cross integration, as of \int_a^b B^{''}_k*B^{''}_l dt '''
        if (i_start, i_start2, f_start, f_end) in self.d_cache_crsint.keys():
            pass
        else:
            f_term1 = self.splder(i_start, 3, f_end  , 1) * self.splder(i_start2, 3, f_end,   2)
            f_term2 = self.splder(i_start, 3, f_start, 1) * self.splder(i_start2, 3, f_start, 2)
            ls_windows = [f_start] + [f_time for f_time in self.ls_knots if f_start < f_time < f_end] + [f_end]
            f_term3 = sum(self.splder(i_start2, 3, ls_windows[j-1], 3) * (self.splrep(i_start, 3, ls_windows[j])- self.splrep(i_start, 3, ls_windows[j-1])) for j in range(1, len(ls_windows)))
            self.d_cache_crsint[(i_start, i_start2, f_start, f_end)] = f_term1 - f_term2 - f_term3
        return self.d_cache_crsint[(i_start, i_start2, f_start, f_end)]


def main():
	test_splrep()
	test_splev()
	test_splint()


if __name__ == '__main__':
	main()
