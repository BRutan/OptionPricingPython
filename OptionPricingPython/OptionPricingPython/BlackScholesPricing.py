#############################################
## BlackScholesPricing.py
#############################################
## Description:
## * Computes the price of a European stock option
## using Black-Scholes formula.

import itertools
import math as m
import numpy
from scipy.stats import norm
import matplotlib.pyplot as plotter
from mpl_toolkits.mplot3d import Axes3D
import threading

__all__ = ['BlackScholes']

class BlackScholes(object):
    """
    * Computes the price of a European stock option
    using Black-Scholes formula.
    """
    # All valid y arguments for plotter (must be one of):
    __yArgsValid = {'price' : True, 'delta' : True, 'gamma' : True, 'vega' : True, 'rho' : True, 'theta' : True, 'expectedreturn' : True, 'optionvol' : True}
    # Valid x arguments for plotter (variable -> (StartVal, EndVal):
    __xArgsPlotting = {'k' : (0, 0), 'q' : (0, 0), 'r' : (0, 0), 's' : (0, 0), 'sigma' : (0, 0), 't' : (0, 0)}
    __xArgsConstructor = {'k' : 0, 'q' : 0, 'r' : 0, 's' : 0, 't' : 0, 'sigma' : 0, 'type' : ''}
    def __init__(self, args):
        """
        * Construct object to compute price of European stock option
        under risk-neutral pricing framework.
        Inputs:
            * args: Expecting dictionary containing all of the following keys:
            {
            * r: Yearly risk-free rate (numeric).
            * sigma: Yearly standard deviation of prices (numeric, > 0).
            * strike: Strike price for option (numeric, >= 0).
            * s: Starting price (numeric, >= 0).
            * q: Continuous dividend rate (numeric, >= 0).
            * T: Years until expiry (numeric, > 0).
            * type: 'call' or 'put' (String, case insensitive).
            }
        """
        self.__reqArgs = BlackScholes.__xArgsConstructor.copy()
        for arg in self.__reqArgs:
            self.__reqArgs[arg] = False
        
        BlackScholes.__ValidateAndSetConstructor(self, args)
    ###################################
    # Properties:
    ###################################
    @property
    def S_0(self):
        return self.__s_0
    @property
    def Strike(self):
        return self.__strike
    @property
    def DivRate(self):
        return self.__divRate
    @property
    def RiskFree(self):
        return self.__riskFree
    @property
    def Sigma(self):
        return self.__sigma
    @property
    def Type(self):
        return self.__type
    @property 
    def T(self):
        return self.__expiry
    @property
    def D_1(self):
        """
        * Calculate and return d_1 used in option pricing.
        """
        d_1 = m.log(self.S_0/self.Strike) + (self.RiskFree - self.DivRate + .5 * self.Sigma * self.Sigma) * self.T
        return d_1 / (self.Sigma * m.sqrt(self.T))
    @property
    def D_2(self):
        """
        * Calculate and return d_2 used in option pricing.
        """
        return self.D_1 - self.Sigma * m.sqrt(self.T)
    @property
    def Price(self):
        """
        * Compute price of option.
        """
        r = self.__riskFree
        s_0 = self.__s_0
        k = self.__strike
        T = self.__expiry
        q = self.__divRate
        sig = self.__sigma
        d_1 = self.D_1
        d_2 = self.D_2
        price = m.exp(-q * T) * s_0 * norm.cdf(d_1) - m.exp(-r * T) * k * norm.cdf(d_2)
        if self.__type == 'call':
            # Return price of call using Black-Scholes formula:
            return price
        else:
            # Compute put option price using put-call parity:
            return price + k * m.exp(-r * T) - s_0 * m.exp(-q * T)
    @property
    def Delta(self):
        """
        * Return Delta of option (linear change in price wrt S_0).
        """
        q = self.__divRate
        T = self.__expiry
        d_1 = self.D_1
        delta = m.exp(-q * T) * norm.cdf(d_1)
        if self.__type == 'call':
            return delta
        else:
            # Calculate delta using put-call parity:
            return delta - m.exp(-q * T)
    @property
    def Gamma(self):
        """
        * Return Gamma of option (second order derivative wrt S_0).
        """
        s_0 = self.__s_0
        T = self.__expiry
        d_1 = self.D_1
        sig = self.__sigma
        # Option gamma is same for puts and calls:
        return norm.pdf(d_1) / (s_0 * sig * m.sqrt(T))
    @property
    def Rho(self):
        """
        * Calculate the Rho of option (linear change in price wrt risk-free rate).
        """
        k = self.__strike
        T = self.__expiry
        r = self.__riskFree
        d_2 = self.D_2
        rho = k * T * m.exp(-r * T)
        if self.__type == 'call':
            return rho * norm.cdf(d_2)
        else:
            return -rho * norm.cdf(-d_2)
    @property
    def Theta(self):
        """
        * Calculate the Theta of option (first order derivative wrt T).
        """
        q = self.__divRate
        s_0 = self.__s_0
        T = self.__expiry
        r = self.__riskFree
        k = self.__strike
        sig = self.__sigma
        d_1 = self.D_1
        d_2 = self.D_2
        theta = -m.exp(-q * T) * s_0 * norm.pdf(d_1) * sig / (2 * m.sqrt(T))
        if self.__type == 'call':
            theta -= r * k * m.exp(-r * T) * norm.cdf(d_2)
        else:
            # Calculate theta using put-call parity:
            theta += r * k * m.exp(-r * T) * norm.cdf(-d_2)
        return theta
    @property
    def Vega(self):
        """
        * Calculate the Vega of option (linear change in price wrt sigma).
        """
        # Vega is the same for puts and calls:
        q = self.__divRate
        s_0 = self.__s_0
        T = self.__expiry
        d_1 = self.D_1
        return s_0 * m.exp(-q * T) * norm.pdf(d_1) * m.sqrt(T)
    @property
    def OptionVol(self):
        """
        * Return option volatility.
        """
        return self.__s_0 * self.Delta * self.__sigma / self.Price
    @property
    def ParamsString(self):
        """
        * Return parameters in string format.
        """
        args = [self.GetProperty('type')]
        for arg in BlackScholes.__xArgsConstructor:
            val = self.GetProperty(arg)
            if isinstance(val, int):
                args.append(arg + ":" + str(int(val)))
            elif isinstance(val, float):
                args.append(arg + ":{0:.2f}".format(val))
        return ','.join(args)
    @property
    def Greeks(self):
        """
        * Return dictionary containing all greeks.
        """
        return {'Delta:' : self.Delta, 'Gamma:': self.Gamma, 'Rho:' : self.Rho, 'Vega:': self.Vega, 'Theta:' : self.Theta}
    @property
    def AttributeString(self):
        greeks = self.Greeks
        strs = [''.join(['European {', self.ParamsString, '}'])]
        for key in greeks.keys():
            strs.append(''.join([key, ':', str(greeks[key])]))
        strs.append(''.join(['ExpectedReturn:', str(self.ExpectedReturn())]))
        strs.append(''.join(['Volatility:', str(self.OptionVol)]))

        return '\n'.join(strs)
    ###################################
    # Setters:
    ###################################
    @S_0.setter
    def S_0(self, s_0):
        if not isinstance(s_0, float) and not isinstance(s_0, int):
            raise Exception("S_0 must be numeric.")
        elif s_0 <= 0:
            raise Exception("S_0 must be positive.")
        self.__reqArgs["s"] = True
        self.__s_0 = s_0
    @Strike.setter
    def Strike(self, strike):
        if not isinstance(strike, float) and not isinstance(strike, int):
            raise Exception("Strike must be numeric.")
        elif strike < 0:
            raise Exception("Strike must be non-negative.")
        self.__reqArgs["k"] = True
        self.__strike = strike
    @DivRate.setter
    def DivRate(self, divRate):
        if not isinstance(divRate, float) and not isinstance(divRate, int):
            raise Exception("DivRate must be numeric.")
        elif divRate < 0:
            raise Exception("DivRate must be non-negative")
        self.__reqArgs["q"] = True
        self.__divRate = divRate
    @RiskFree.setter
    def RiskFree(self, riskFree):
        if not isinstance(riskFree, float) and not isinstance(riskFree, int):
            raise Exception("riskFree must be numeric.")
        self.__reqArgs["r"] = True
        self.__riskFree = riskFree
    @Sigma.setter
    def Sigma(self, sigma):
        if not isinstance(sigma, float) and not isinstance(sigma, int):
            raise Exception("sigma must be numeric.")
        elif sigma <= 0:
            raise Exception("sigma must be positive.")
        self.__reqArgs["sigma"] = True
        self.__sigma = sigma
    @Type.setter
    def Type(self, type_in):
        if not isinstance(type_in, str):
            raise Exception("type must be a string.")
        elif type_in.lower() != 'call' and type_in.lower() != 'put':
            raise Exception("type must be 'call' or 'put'.")
        self.__reqArgs["type"] = True
        self.__type = type_in.lower()
    @T.setter
    def T(self, T_in):
        if not isinstance(T_in, float) and not isinstance(T_in, int):
            raise Exception("T must be numeric.")
        elif T_in <= 0:
            raise Exception("T must be positive.")
        self.__reqArgs["t"] = True
        self.__expiry = T_in

    ###################################
    # Interface Methods:
    ###################################    
    def InstantaneousChg(self, mu):
        """
        * Compute instantaneous change in price of option, given risk free rate (mu).
        Inputs:
        * mu: Expecting numeric value, or None. If not specified, then uses current
        risk-free rate.
        """
        if not mu:
            mu = self.RiskFree
        if not isinstance(mu, int) and not isinstance(mu, float):
            raise Exception("mu must be numeric.")
        r_orig = self.RiskFree
        sig = self.__sigma 
        s = self.__s_0
        self.RiskFree = mu
        chg = .5 * (( sig * s )** 2) * self.Gamma + mu * s * self.Delta + self.Theta
        # Reset the risk-free rate to original value:
        self.RiskFree = r_orig
        return chg
        
    def ExpectedReturn(self, mu = None):
        """
        * Compute expected return of option, given risk free rate (mu).
        Inputs:
        * mu: Expecting numeric value, or None. If not specified, then uses current
        risk-free rate.
        """
        if mu and not isinstance(mu, int) and not isinstance(mu, float):
            raise Exception("mu must be numeric.")
            
        return self.InstantaneousChg(mu) / self.Price

    def PlotRelationships(self, yArg, xArgs, numPts = 100):
        """
        * Plot one relationship of 
        y = [ Price, Delta, Gamma, Vega, Rho, Theta, ExpectedReturn, OptionVolatility] (one, case insensitive)
        vs x = [ s, r, q, t, k, sigma ] (>= 1, case insensitive).
        Inputs:
        * yArg: Expecting string denoting which y variable to map x values to.
        * xArgs: Expecting { xStr -> tuple() } mapping of { one/more of x listed above -> (StartVal, EndVal, #Points) }.
        * numPts: Expecting integer denoting # of points to generate grid with. 100 by default. > 100 will result
        in significantly slower speeds.
        """
        # Ensure all parameters were valid, throw exception if not:
        BlackScholes.__ValidatePlotting(yArg, xArgs, numPts)
        yArg = BlackScholes.__ConvertYArg(yArg)
        # Plot data for each sensitivity:
        data = {}
        xArgNames = []
        # Generate all x data:
        for arg in xArgs.keys():
            origVal = self.GetProperty(arg)
            xArgNames.append([arg, origVal])
            data[arg] = []
            startVal = xArgs[arg][0]
            endVal = xArgs[arg][1]
            stepSize = (endVal - startVal) / numPts
            currStep = startVal
            while currStep <= endVal:
                data[arg].append(currStep)
                currStep += stepSize
            # Handle floating point issues:
            while len(data[arg]) > numPts:
                data[arg].pop()
        # If doing multidimensional plot, generate all cross varying maps to y given
        # all possible 2 combinations of x dimensions:
        xParams = list(xArgs.keys())
        combins = []
        if len(xParams) > 1:
            # Get all combinations of 2 X parameters
            for subset in itertools.combinations(xParams, 2):
                combins.append(subset)
        else:
            combins.append([xParams[0]])        
        # Generate all y data given x data (or all possible 2-combinations thereof):
        pt = 0
        meshes = {}
        if len(xArgNames) > 1:
            # Generate meshes for each combination of parameters, and mesh for output:
            for combin in combins:
                xArg1 = combin[0]
                xArg2 = combin[1]
                combinKey = xArg1 + xArg2
                X_1, X_2 = numpy.meshgrid(data[xArg1], data[xArg2])
                Y = numpy.zeros((numPts, numPts))
                row = 0
                # Calculate all y values using mesh values:
                while row < numPts:
                    col = 0
                    while col < numPts:
                        self.SetProperty(xArg1, X_1[row][col])
                        self.SetProperty(xArg2, X_2[row][col])
                        Y[row][col] = getattr(self, yArg)
                        col += 1
                    row += 1
                meshes[combinKey] = (X_1, X_2, Y)
        else:
            # Generate y value for every x value for 2D plot:
            xArg = xArgNames[0][0]
            Y = []
            X = data[xArg]
            while pt < numPts:
                self.SetProperty(xArg, data[xArg][pt])
                Y.append(getattr(self, yArg))
                pt += 1
            meshes[xArg] = (X, Y)

        # Reset all x argument values to original values:
        for vals in xArgNames:
            self.SetProperty(vals[0], vals[1])
        ##################
        # Plot all of the sensitivities:
        ##################
        plotObj = plotter.figure()
        title = ''.join(['European {', self.ParamsString, '}'])
        if len(xArgs.keys()) > 1:
            for combin in combins:
                # Use 3-D plot:
                combinKey = combin[0] + combin[1]
                X_1, X_2, Y = meshes[combinKey]
                axes = plotObj.add_subplot(111, projection='3d')
                axes.plot_wireframe(X_1, X_2, Y)
                axes.set_xlabel(combin[0])
                axes.set_ylabel(combin[1])
                axes.set_zlabel(yArg)
                axes.title.set_text(title)
        else:
            # Use 2-D plot:
            xArg = xArgNames[0][0]
            X, Y = meshes[xArg]
            plotObj.suptitle(title, fontsize = 10)
            axis = plotObj.add_subplot('111')
            axis.plot(X, Y)
            axis.set_ylabel(yArg)
            axis.set_xlabel(xArg)
        plotObj.show()

        return plotObj

    def PrintAttributes(self):
        """
        * Print all option attributes (price info, greeks, etc) to stdout.
        """
        print(self.AttributeString)
    
    ###################################
    # Static Helpers:
    ###################################
    @staticmethod
    def XArgsPlotting():
        """
        * Return copy x arg map for use in plotting.
        """
        return BlackScholes.__xArgsPlotting.copy()
    @staticmethod
    def RequiredConstructorArgs():
        """
        * Return copy of all required arguments for this class' constructor.
        """
        return BlackScholes.__xArgsConstructor.copy()
    ###################################
    # Private Helpers:
    ###################################
    def __ValidateAndSetConstructor(self, args):
        """
        * Validate all passed parameters to the constructor. Raise 
        exception if any are invalid. Set if acceptable.
        """
        errMsgs = []
        invalidArgs = []
        # Validate all passed parameters:
        for arg in args.keys():
            try:
                _lower = str(arg).lower()
                if _lower not in self.__reqArgs.keys():
                    invalidArgs.append(arg)
                elif _lower == 'k':
                    self.Strike = args[arg]
                elif _lower == 'q':
                    self.DivRate = args[arg]
                elif _lower == 'r':
                    self.RiskFree = args[arg]
                elif _lower == 's':
                    self.S_0 = args[arg]
                elif _lower == 'sigma':
                    self.Sigma = args[arg]
                elif _lower == 't':
                    self.T = args[arg]
                elif _lower == 'type':
                    self.Type = args[arg]
            except Exception as ex:
                errMsgs.append(ex.message)
        # List all invalid arguments (not in the required arguments dictionary):
        if len(invalidArgs) > 0:
            errMsgs.Append(''.join(['The following args were invalid:', ','.join(invalidArgs)]))

        # Ensure all required parameters were passed:
        missingArgs = []
        for arg in self.__reqArgs.keys():
            if self.__reqArgs[arg] == False:
                missingArgs.append(arg)
                
        if len(missingArgs) > 0:
            errMsgs.append(''.join(['The following required args were missing:', ','.join(missingArgs)]))
        # Raise exception if any parameters were invalid:
        if len(errMsgs) > 0:
            raise Exception('\n'.join(errMsgs))

    @staticmethod
    def __ConvertYArg(yArg):
        """
        * Convert case insensitive y argument to case sensitive version to allow use with
        getattr().
        {'price' : True, 'delta' : True, 'gamma' : True, 'vega' : True, 'rho' : True, 'theta' : True, 'expectedreturn' : True, 'optionvol' : True}
        """
        if yArg == 'price':
            return 'Price'
        if yArg == 'delta':
            return 'Delta'
        if yArg == 'gamma':
            return 'Gamma'
        if yArg == 'vega':
            return 'Vega'
        if yArg == 'rho':
            return 'Rho'
        if yArg == 'theta':
            return 'Theta'
        if yArg == 'expectedreturn':
            return 'ExpectedReturn'
        if yArg == 'optionvol':
            return 'OptionVol'

    @staticmethod
    def __ValidatePlotting(yArg, xArgs, numPts):
        """
        * Validate all input arguments to the plotting method.
        """
        invalidArgs = []
        messages = []
        # Ensure that input arguments are valid:
        if not isinstance(yArg, str):
            invalidArgs.append("yArg must be a string.")
        elif not yArg.lower() in BlackScholes.__yArgsValid.keys():
            validYArgs = ','.join(list(BlackScholes.__yArgsValid.keys()))
            temp = ["yArg {", yArg.lower(), "} must be one of [", validYArgs, "]"]
            invalidArgs.append(''.join(temp))
        if not isinstance(numPts, int) and not isinstance(numPts, float):
            invalidArgs.append("numPts must be numeric.")
        if not isinstance(xArgs, dict):
            invalidArgs.append("xArgs must be a dictionary.")
        elif len(xArgs.keys()) == 0:
            invalidArgs.append("xArgs must have at least one key.")
        else:
            # Ensure that tuples of correct dimension were provided for each argument:
            invalidMap = []
            invalidXArgs = []
            for arg in xArgs.keys():
                _arg = arg.lower()
                tup = xArgs[arg]
                if _arg not in BlackScholes.__xArgsPlotting.keys():
                    invalidXArgs.append(_arg)
                elif not isinstance(tup, tuple):
                    invalidMap.append(arg)
                elif len(tup) != 2:
                    invalidMap.append(arg)
            if len(invalidXArgs) > 0:
                invalidXArgs = ['The following xArgs keys are invalid:{', ','.join(invalidXArgs), '}']
                invalidArgs.append(''.join(invalidXArgs))    
            if len(invalidMap) > 0:
                invalidMap = ['The following xArgs were not mapped to tuples of length 2:{', ','.join(invalidMap), '}']
                invalidArgs.append(''.join(invalidMap))
      
        if len(invalidArgs) > 0:
            raise Exception('\n'.join(invalidArgs))

    def SetProperty(self, arg, val):
        """
        * Set the requested property using string and value.
        Inputs:
        * arg: Expecting a string.
        * val: Expecting a numeric value or a string (if for type).
        """
        arg = str(arg).lower()
        if arg == 'k':
            self.Strike = val
        elif arg == 'q':
            self.DivRate = val
        elif arg == 'r':
            self.RiskFree = val
        elif arg == 's':
            self.S_0 = val
        elif arg == 'sigma':
            self.Sigma = val
        elif arg == 't':
            self.T = val
        elif arg == 'type':
            self.Type = val
        else:
            raise Exception(arg + ' is invalid.')

    def GetProperty(self, arg):
        """
        * Return requested property using string.
        """
        arg = str(arg).lower()
        if arg == 'k':
            return self.Strike
        if arg == 'q':
            return self.DivRate
        if arg == 'r':
            return self.RiskFree
        if arg == 's':
            return self.S_0
        if arg == 'sigma':
            return self.Sigma
        if arg == 't':
            return self.T
        if arg == 'type':
            return self.Type
        else:
            raise Exception(arg + ' is invalid.')