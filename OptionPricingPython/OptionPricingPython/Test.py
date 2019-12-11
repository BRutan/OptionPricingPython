
from BlackScholesPricing import BlackScholes
from BinomialPricing import BinomialOptionModel

if __name__ == '__main__':
    # Test the binomial option model:
    model = BinomialOptionModel(0, .3, 100, 98, 10, .25, 'call')
    model.ConstructTree()
    # Plot the hedge ratios for each period (of Stock-Call replicating portfolio):
    plot = model.PlotOutputs('hr_stockcall')
    plot.close()
    # Plot the hedge ratios for each period (of Stock-Bond replicating portfolio):
    plot = model.PlotOutputs('hr_stockbond')
    plot.close()

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
    
    # Plot relationships between option volatility and underlying implied volatility:
    # Start with strike = 50 (deep ITM):
    plotArgs = BlackScholes.XArgsPlotting()
    copy = plotArgs.copy()
    for arg in copy.keys():
        if arg != 'sigma':
            del plotArgs[arg]
    plotArgs['sigma'] = (.3, .5)
    bsModel.Strike = 50
    plot1 = bsModel.PlotRelationships("optionvol", plotArgs)

    # Do strike = 200 (far OTM):
    bsModel.Strike = 200
    plot2 = bsModel.PlotRelationships("optionvol", plotArgs)
    
    plot1.close()
    plot2.close()
    print("pause")