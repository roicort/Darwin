import scipy as sc
import numpy as np

###############################################################################################

ackleybounds       = [-30, 30]
dixonpricebounds   = [-10, 10]
griewankbounds     = [-600, 600]
levybounds         = [-10, 10]
michalewiczbounds  = [0, sc.pi]
permbounds         = ["-dim", "dim"]  # min at [1 2 .. n]
powellbounds       = [-4, 5]  # min at tile [3 -1 0 1]
powersumbounds     = [0, "dim"]  # 4d min at [1 2 3 4]
rastriginbounds    = [-5.12, 5.12]
rosenbrockbounds   = [-2.4, 2.4]  # wikipedia
schwefelbounds     = [-500, 500]
spherebounds       = [-5.12, 5.12]
sum2bounds         = [-10, 10]
tridbounds         = ["-dim**2", "dim**2"]  # fmin -50 6d, -200 10d
zakharovbounds     = [-5, 10]
ellipsebounds      =  [-2, 2]
logsumexpbounds    = [-20, 20]  # ?
nesterovbounds     = [-2, 2]
powellsincosbounds = [ "-20*pi*dim", "20*pi*dim"]
randomquadbounds   = [-10000, 10000]
saddlebounds       = [-3, 3]

###############################################################################################

#https://github.com/denis-bz

def ackley( x, a=20, b=0.2, c=2*sc.pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( sc.cos( c * x ))
    return -a*sc.exp( -b*sc.sqrt( s1 / n )) - sc.exp( s2 / n ) + a + sc.exp(1)

#...............................................................................
def dixonprice( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

#...............................................................................
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = sc.prod( sc.cos( x / sc.sqrt(j) ))
    return s/fr - p + 1

#...............................................................................
def levy( x ):
    x = np.asarray_chkfinite(x)
    z = 1 + (x - 1) / 4
    return (sc.sin( sc.pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sc.sin( sc.pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sc.sin( 2 * sc.pi * z[-1] )**2 ))

#...............................................................................
def michalewicz( x ):  # mich.m
    michalewicz_m = .5  # orig 10: ^20 => underflow
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sc.sin(x) * sc.sin( j * x**2 / sc.pi ) ** (2 * michalewicz_m) )

#...............................................................................
def perm( x, b=.5 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return sc.mean([ sc.mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])
    # original overflows at n=100 --
    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
    #       for k in j ])

#...............................................................................
def powell( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = sc.sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) **2
    f[3] = sc.sqrt(10) * (x[0] - x[3]) **2
    return sum( f**2 )

#...............................................................................
def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

#...............................................................................
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * sc.cos( 2 * sc.pi * x ))

#...............................................................................
def rosenbrock( x ):  # rosen.m
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

#...............................................................................
def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * sc.sin( sc.sqrt( abs( x ))))

#...............................................................................
def sphere( x ):
    x = np.asarray_chkfinite(x)
    return sum( x**2 )

#...............................................................................
def sum2( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * x**2 )

#...............................................................................
def trid( x ):
    x = np.asarray_chkfinite(x)
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

#...............................................................................
def zakharov( x ):  # zakh.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4

#...............................................................................
def ellipse( x ):
    x = np.asarray_chkfinite(x)
    return sc.mean( (1 - x) **2 )  + 100 * sc.mean( np.diff(x) **2 )

#...............................................................................
def nesterov( x ):
    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))

#...............................................................................
def saddle( x ):
    x = np.asarray_chkfinite(x) - 1
    return np.mean( np.diff( x **2 )) \
        + .5 * np.mean( x **4 )

###############################################################################################
