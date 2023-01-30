import ib_insync as ibi
'''
'STK' = Stock (or ETF)
'OPT' = Option
'FUT' = Future
'IND' = Index
'FOP' = Futures option
'CASH' = Forex pair
'CFD' = CFD
'BAG' = Combo
'WAR' = Warrant
'BOND' = Bond
'CMDTY' = Commodity
'NEWS' = News
'FUND' = Mutual fund
'CRYPTO' = Crypto currency
'''
options = list()
options = ['NVDA', 'AAPL', 'NFLX', 'MRNA', 'VKTX', 'TSLA','MDGL',
                         'FUTU', 'AMZN', 'NOW', 'IWM', 'MDB', 'QQQ', 'FSLR',
                         'GOOGL', 'SNOW']
list = list()
stocks = ['TSLA', 'MSFT', 'AAPL', 'INTC', 'GOOG', 'META', 'RIVN', 'LCID', 'ZM', 'TEAM', 'EBAY', 'QCOM', 'AVGO', 'ADBE', 'AMGN', 'AMAT', 'ISRG', 'PYPL', 'INTU', 'AMD', 'GILD', 'ADI', 'REGN']
'''
    Map the security type name to a class object.
'''
symbols = ['STK', 'OPT', 'FUT', 'IND', 'FOR', 'CASH']
securityTypeMap  = dict(STK=stocks, OPT=options)
ContractClasses = dict(STK=ibi.Stock, OPT=ibi.Option, FUT=ibi.Future, IND=ibi.Index,
                     FOP=ibi.FuturesOption, CASH=ibi.Forex, CFD=ibi.CFD, BAG=ibi.Bag,
                     WAR=ibi.Warrant, BOND=ibi.Bond, CMDTY=ibi.Commodity,
                     FUND=ibi.MutualFund, CRYPTO=ibi.Crypto)
                     # 'NEWS':Stock,
# Mapping of symbols to attributes:
#                SecType, exchange, currency

# print(f"Options' {len(options['Securities'])} Symbols: {options['Securities']}")
class Bundle():
    '''
        Holds the set of securities in play and ops to manage them.
        Each set may include as many as 50 securities as a single entity
        to be tracked, averaged, watched for triggering events
    '''

    def __init__(self, secType="OPT", exchange="", currency="USD"):
        '''
            Setup all of the info needed to create a contract for each Symbol.
            Each security requires a contract to bind that symbol to the data
            stream and each contract needs to know what exchange the security
            operates in and the currency denomination of trades.
        '''

        if not secType in ContractClasses:
            secType = 'OPT'

        self.bundle = dict()
        self.bundle['Class'] = ContractClasses[secType]
        self.bundle['Exchange'] = exchange
        self.bundle['Currency'] = currency
        self.bundle['Securities'] = securityTypeMap[secType]


    def list(self):
        '''
            Return the list of securities the bundle represents.
        '''
        ## print(f"bundle[Securities] {self.bundle['Securities']}")
        return self.bundle['Securities']

    async def register(self, security):
        '''
            Set the data flowing. Set the contract and return the Tick
        '''
        if not security in self.bundle['Securities']:
            print(f'{security} is not represneted by this bundle.')

        await asyncio.sleep(0)
        return None


    async def run(ib, sym_ticks):
        """"
            Monitor the data stream and process each tick.
        """
        async for tickers in ib.pendingTickersEvent:
            for ticker in tickers:
                _tick = sym_ticks[ticker.contract.symbol]
                await asyncio.sleep(0)
                # logger.debug(ticker)
                # each Ticks object will see all subscriptions
                # first check for redundant ticks
                if (not np.isnan(ticker.volume) and not ticker.volume == 0):
                    _tick.latest_volume = ticker.volume
                    _tick.queued_tickers.append(ticker)
                    q_len = len(_tick.queued_tickers)
                    if q_len > 10:
                        logger.debug(
                            f"queued {ticker.contract.symbol}," f" queue len: {q_len}"
                        )
                # else:
                #     logger.debug(
                #         f"tossed non-matching ticker,"
                #         f" queue len: {len(self.queued_tickers)}"
                #     )
    
                # can only return once per call, so we can get backed up
                # use "bad" ticker events to help drain the queue
                if len(_tick.queued_tickers) > 0:
                    """
                        If this particular ticker stream (subscription) actually contains
                        ticks then pop the oldest from the queue and return it.
                    """
                    ticker_ = _tick.queued_tickers.popleft()
                    await asyncio.sleep(0)  # printing and scrolling is slow
                    return ticker_
                else:
                    """
                        The queue hasn't any elements so return None.
                        Async calls always return some value else async gather won't finish.
                    """
                    return None
