
from BlackScholesPricing import BlackScholes
from BinomialPricing import BinomialOptionModel
import csv

if __name__ == '__main__':
    # Test the binomial option model:
    #model = BinomialOptionModel(0, .3, 100, 98, 10, .25, 'call')
    #model.ConstructTree()
    # Plot the hedge ratios for each period (of Stock-Call replicating portfolio):
    #plot = model.PlotOutputs('hr_stockcall')
    # Plot the hedge ratios for each period (of Stock-Bond replicating portfolio):
    #plot = model.PlotOutputs('hr_stockbond')

    #########################
    # Test the Black Scholes model:
    #########################
    args = BlackScholes.RequiredConstructorArgs()
    
    args['k'] = 98
    args['q'] = 0
    args['r'] = 0
    args['s'] =  100
    args['sigma'] = .3
    args['t'] = .25
    args['type'] = 'call'
    
    bsModel = BlackScholes(args)
    # Compute the expected return of the option (given parameters):
    print(''.join(['European {', bsModel.ParamsString, '}']))
    print("Expected return:")
    print(bsModel.ExpectedReturn(.1))
    print("Price")
    print(bsModel.Price)
    
    # Plot relationships between option volatility and underlying implied volatility:
    # Start with strike = 50 (deep ITM):
    plotArgs = BlackScholes.XArgsPlotting()
    copy = plotArgs.copy()
    for arg in copy.keys():
        if arg != 't' and arg != 'sigma':
            del plotArgs[arg]
    plotArgs['t'] = (.01, .02)
    plotArgs['sigma'] = (.3, .5)
    bsModel.Strike = 100
    obj = bsModel.PlotRelationships("vega", plotArgs, 100)
    # Do strike = 200 (far OTM):
    bsModel.Strike = 200
    obj = bsModel.PlotRelationships("optionvol", plotArgs, numPts = 100)
    
    print("pause")