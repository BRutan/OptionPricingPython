#############################################
## BinomialPricing.py
#############################################
## Description:
## * Price option using standard binomial tree method.

import math as m
import operator as op
from functools import reduce
import matplotlib.pyplot as plt

__all__ = ['BinomialOptionModel', 'BinomNode']

class BinomialOptionModel(object):
    """
    * Models a standard binomial stock option tree with default
    up and down probabilities (based upon risk-neutral pricing).
    """
    __plotArgs = {'hr_stockcall' : True, 'hr_stockbond' : True, 'optionprice' : True, 'assetprice' : True}
    def __init__(self, riskFree, sigma, s_0, strike, nSteps, T, type):
        """
        * Construct new standard binomial option tree.
        Inputs:
            * riskFree: Yearly risk-free rate (Floating Point).
            * sigma: Yearly standard deviation of prices (Floating point, > 0).
            * strike: Strike price for option (Floating point, >= 0).
            * s_0: Starting price (Floating point, > 0).
            * nSteps: Number of steps in the model (Integer, > 0).
            * T: Years until expiry (Floating Point, > 0).
            * type: 'call' or 'put' (String, case insensitive).
        """
        # Ensure all parameters are valid: 
        BinomialOptionModel.__Validate(riskFree, sigma, s_0, strike, nSteps, T, type)
        # Map { Step # -> { # Ups -> Node }}:
        self.__nodes = {}
        self.__underlyingStart = s_0
        self.__strike = strike
        self.__riskFree = riskFree
        self.__sigma = sigma
        self.__steps = nSteps
        self.__expiry = T
        self.__type = type.lower()
        # Use standard u, d and Prob(u) from risk neutral pricing:
        self.__timeStep = self.__expiry / self.__steps
        self.__u = m.exp(self.__sigma * m.sqrt(self.__timeStep))
        self.__d = 1 / self.__u
        self.__pUp = ((1 + self.__riskFree / self.__steps) ** (self.__timeStep) - self.__d) / (self.__u - self.__d)

    def ConstructTree(self):
        """
        * Construct all the nodes in the tree.
        """
        step = 0
        totalNodes = 0
        while step <= self.__steps:
            self.__nodes[step] = {}
            nUps = 0
            while nUps <= totalNodes:
                combins = BinomialOptionModel.__nCr(totalNodes, nUps)
                self.__nodes[step][nUps] = BinomNode(self.__underlyingStart, nUps, totalNodes - nUps, step, combins)
                nUps += 1
            totalNodes += 1
            step += 1
        # Price the option at each node:
        self.__CalcOptionPrices()
        # Determine asset prices at each node:
        self.__CalcAssetPrices()
        # Compute all the hedge ratios at each node:
        self.__ComputeSCHRs()
        # Compute all stock + bond replicating portfolio hedge ratios at each node:
        self.__ComputeSBHRs()
    ###################################
    # Properties:
    ###################################
    @property
    def AssetPrices(self):
        return self.__underlyingPrices
    @property
    def OptionPrices(self):
        return self.__optPrices
    @property
    def StockBondHedgeRatios(self):
        return self.__HRSB
    @property
    def StockCallHedgeRatios(self):
        return self.__HRSC
    ###################################
    # Private Helpers:
    ###################################
    def __ComputeSBHRs(self):
        """
        * Return hedge ratios of Bond + Stock portfolio at step {step# -> {#Ups -> (StockHR, BondHR)}}.
        """
        step = self.__steps - 1
        hedgeRatios = {}
        u = self.__u
        d = self.__d
        while step >= 0:
            currHRs = {}
            time = self.__expiry * (self.__steps - step) / self.__steps
            for nUps in self.__nodes[step].keys():
                upNode = self.__nodes[step + 1][nUps + 1]
                downNode = self.__nodes[step + 1][nUps]
                upAssetPrice = upNode.AssetPrice(u, d)
                downAssetPrice = downNode.AssetPrice(u, d)
                stockHR = (upNode.OptionPrice - downNode.OptionPrice) / (upAssetPrice - downAssetPrice) 
                bondHR = (downNode.OptionPrice * upAssetPrice - upNode.OptionPrice * downAssetPrice) / (upAssetPrice - downAssetPrice)
                currHRs[nUps] = ((stockHR, bondHR))
            hedgeRatios[step] = currHRs
            step -= 1
        self.__HRSB = hedgeRatios

    def __ComputeSCHRs(self):
        """
        * Return hedge ratios of Call + Stock portfolio at step {step# -> {#Ups -> (StockHR, CallHR)}}.
        """
        step = self.__steps - 1
        hedgeRatios = {}
        u = self.__u
        d = self.__d
        probDiff = u - d
        while step >= 0:
            currHRs = {}
            for nUps in self.__nodes[step].keys():
                upNode = self.__nodes[step + 1][nUps + 1]
                downNode = self.__nodes[step + 1][nUps]
                stockHR = (upNode.OptionPrice - downNode.OptionPrice) / (upNode.AssetPrice(u, d) - downNode.AssetPrice(u, d)) 
                callHR = 1
                currHRs[nUps] = ((stockHR, callHR))
            hedgeRatios[step] = currHRs
            step -= 1
        self.__HRSC = hedgeRatios

    def __CalcAssetPrices(self):
        """
        * Fill dictionary containing all underlying prices.
        """
        self.__underlyingPrices = {}
        u = self.__u
        d = self.__d
        step = 0
        maxSteps = max(list(self.__nodes.keys()))
        while step <= maxSteps:
            self.__underlyingPrices[step] = {}
            for nUps in self.__nodes[step].keys():
                node = self.__nodes[step][nUps]
                self.__underlyingPrices[step][nUps] = node.AssetPrice(u, d)
            step += 1

    def __CalcOptionPrices(self):
        """
        * Calculate the price of the option at each node in the tree using risk-neutral pricing.
        """
        # Exit if ConstructTree() has not been called yet.
        if len(self.__nodes.keys()) == 0:
            return
        self.__optPrices = {}
        # Start with the terminal step:
        step = self.__steps
        pUp = self.__pUp
        pDown = 1 - pUp
        while step >= 0:
            time = self.__expiry * (self.__steps - step) / self.__steps
            df = m.exp(-self.__riskFree * time)
            self.__optPrices[step] = {}
            for nUps in self.__nodes[step].keys():
                node = self.__nodes[step][nUps]
                if step == self.__steps:
                    # Compute simple payoff of the option at expiry:
                    node.OptionPrice = self.__Payoff(node.AssetPrice(self.__u, self.__d))
                else:
                    # Compute current option price as discounted expected value of option
                    # under risk-neutral pricing:
                    upNode = self.__nodes[step + 1][nUps + 1]
                    downNode = self.__nodes[step + 1][nUps]
                    node.OptionPrice = (upNode.OptionPrice * pUp + downNode.OptionPrice * pDown) * df
                self.__optPrices[step][nUps] = node.OptionPrice
            step -= 1

    def Nodes(self, stepNum = None):
        """
        * Return all nodes (in { nUps -> Node } that occur at step.
        Inputs:
        * stepNum: Expecting positive integer, less than or equal to total number of steps
        used in model.
        If stepNum is ommitted then 
        """
        # Return all nodes if stepNum not provided:
        if stepNum == None:
            return self.__nodes
        if not isinstance(stepNum, int) and not isinstance(stepNum, float):
            raise Exception("stepNum must be numeric or None.")
        elif stepNum <= 0 or stepNum > self.__numSteps:
            raise Exception("stepNum must be positive, and <= total number of steps.")
        
        # Return nodes at particular step number if requested:
        return self.__nodes[stepNum]

    def PlotOutputs(self, arg):
        """
        * Plot model output at each node in tree-type graph.
        Inputs:
        * arg: Expecting string, one of 'hr_stockcall', 'hr_stockbond', 'optionprice', 'stockprice'.
        """
        hedgeRatio = False
        arg = arg.lower()
        if arg not in BinomialOptionModel.__plotArgs.keys():
            raise Exception(''.join(['arg must be one of {', ','.join(BinomialOptionModel.__plotArgs.keys()), "}"]))
        elif arg == 'hr_stockcall':
            values = self.StockCallHedgeRatios
            title = 'Hedge Ratios [Stock,Call]'
            hedgeRatio = True
        elif arg == 'hr_stockbond':
            title = 'Hedge Ratios [Stock,Bond]'
            values = self.StockBondHedgeRatios
            hedgeRatio = True
        elif arg == 'optionprice':
            title = 'Option Prices'
            values = self.OptionPrices
            hedgeRatio = False
        else:
            title = 'Asset Prices'
            values = self.AssetPrices
            hedgeRatio = False
        ##############################
        # Plot the tree:
        ##############################
        plt.xlim(0,1)
        numSteps = len(values.keys())
        finalNumNodes = len(values[numSteps - 1].keys())
        xStep = 1 / numSteps
        xStart = 0
        yStep = 1 / (finalNumNodes * 2)
        yStart = .5
        currNode = 0
        currStep = 0
        while currStep < numSteps:
            maxUps = max(list(values[currStep].keys()))
            yStart = .5 + yStep * maxUps
            currUp = maxUps
            while currUp >= 0:
                if hedgeRatio:
                    stockWgt = "{0:.2f}".format(values[currStep][currUp][0])
                    bondWgt = "{0:.2f}".format(values[currStep][currUp][1])
                    str = ''.join(["[", stockWgt, ",", bondWgt, "]"])
                else:
                    str = "{0:.2f}".format(values[currStep][currUp])
                plt.figtext(xStart, yStart, str)
                currUp -= 1
                yStart -= yStep * 2
            xStart += xStep
            currStep += 1

        plt.axis('off')
        plt.title(title, loc='left')
        plt.show(block = False)

        return plt

    ###################################
    # Private Helpers:
    ###################################
    def __Payoff(self, underlying):
        """
        * Return payoff of option given passed underlying price.
        """
        if self.__type == 'put':
            return max(0, self.__strike - underlying)
        else:
            return max(0, underlying - self.__strike)
    
    @staticmethod
    def __nCr(n, r):
        """
        * Return number of combinations in efficient manner.
        """
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    @staticmethod
    def __Validate(riskFree, sigma, s_0, strike, nSteps, T, type):
        """
        * Validate all passed parameters to the constructor. Raise 
        exception if any are invalid.
        """
        errMsgs = []
        # Validate all passed parameters:
        if not isinstance(riskFree, float) and not isinstance(riskFree, int):
            errMsgs.append("riskFree must be floating point.")
        if not isinstance(sigma, float) and not isinstance(sigma, int):
            errMsgs.append("sigma must be floating point.")
        elif sigma <= 0:
            errMsgs.append("sigma must be positive.")
        if not isinstance(s_0, float) and not isinstance(s_0, int):
            errMsgs.append("s_0 must be floating point.")
        elif s_0 <= 0:
            errMsgs.append("s_0 must be positive.")
        if not isinstance(strike, float) and not isinstance(strike, int):
            errMsgs.append("strike must be floating point.")
        elif strike < 0:
            errMsgs.append("strike must be non-negative.")
        if not isinstance(nSteps, float) and not isinstance(nSteps, int):
            errMsgs.append("nSteps must be an integer.")
        elif nSteps <= 0:
            errMsgs.append("nSteps must be positive.")
        if not isinstance(T, float) and not isinstance(T, int):
            errMsgs.append("T must be floating point.")
        elif T <= 0:
            errMsgs.append("T must be a positive.")
        if not isinstance(type, str):
            errMsgs.append("type must be a string.")
        elif type.lower() != 'call' and type.lower() != 'put':
            errMsgs.append("type must be 'call' or 'put'.")
        if len(errMsgs) > 0:
            raise Exception('\n'.join(errMsgs))

class BinomNode(object):
    """
    * Node in binomial tree.
    """
    ###############################
    # Constructors:
    ###############################
    def __init__(self, s_0, nUp, nDown, stepNum, combins):
        """
        * Each node contains number of ups, downs,
        and number of paths possible to reach node.
        """
        self.__underlying = s_0
        self.__nUp = nUp
        self.__nDown = nDown
        self.__thisStepCount = stepNum
        # Store the number of combinations of ups and downs
        # possible to reach node, to compute the probability of 
        # reaching the node.
        self.__combins = combins
        self.__optionPrice = 0
    ###############################
    # Accessors:
    ###############################
    @property
    def NUp(self):
        return self.__nUp
    @property
    def NDown(self):
        return self.__nDown
    @property
    def StepNumber(self):
        return self.__thisStepCount
    @property
    def Combins(self):
        return self.__combins
    @property
    def OptionPrice(self):
        return self.__optionPrice
    @OptionPrice.setter
    def OptionPrice(self, price):
        self.__optionPrice = price
    ###############################
    # Interface Functions:
    ###############################
    def PObserve(self, pUp):
        """
        * Return the probability of observing this node.
        """
        return self.__combins * pUp ** self.__nUp * (1 - pUp) ** self.__nDown
    def AssetPrice(self, u, d):
        """
        * Return modeled asset price.
        """
        return self.__underlying * u ** self.__nUp * d ** self.__nDown