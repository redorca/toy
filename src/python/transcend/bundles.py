'''
    Generalize ticker handling through the Bundle() class allowing
    access to methods applicable to all tickers and leave ticker
    specific methods to the class Ticks().

    E.G. the run command applies to all tickers so it moves out of Ticks()
    into Bundle().
'''

import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict

import ib_insync as ibi
from streamer import ticks, candles, emacalc
from streamer import davelogging as dl

logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
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

    def __init__(self, ib_conn, tickers, /, secType="OPT", exchange="", currency="USD"):
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
        self.bundle['Securities'] = tickers
        self.ib = ib_conn
        self.sym_ticks = dict()
        self.candle_maker = dict()
        self.ema_calculator = dict()

    def list(self):
        '''
            Return the list of securities the bundle represents.
        '''
        return self.bundle['Securities']

    async def tick_for_ticker(self, ticker):
        return self.sym_ticks[ticker.contract.symbol]

    async def candle_for_tick(self, tick_):
        return self.candle_maker[tick_.symbol]

    async def register(self, symbols):
        '''
            Set the data flowing. Set the contract and return the Tick
        '''
        dollar_volume = defaultdict(lambda: 100000, {"RSP": 50000})
        self.ema_calculator = emacalc.EmaCalculator()

        # Add the rtTime field to the Ticker.
        tickFields = "233"
        ## for symbol in self.bundle['Securities']:
        for symbol in symbols:
            logger.debug(f"set tick {symbol}")
            tick_src = ticks.Ticks(self.ib, symbol)
            self.sym_ticks[symbol] = tick_src
            self.candle_maker[symbol] = candles.CandleMakerDollarVolume(dollar_volume[symbol])
            self.ib.reqMktData(tick_src.contract,  snapshot=False, genericTickList=tickFields)
            await asyncio.sleep(0)

        return None

    async def isConnected(self):
        await asyncio.sleep(0)
        return self.ib.isConnected()

    async def connection_monitor(self):
        logger.debug("--")
        while True:
            if not self.ib.isConnected():
                logger.debug("Not connected.")
                raise ConnectionError("Connection closed")
                return
            await asyncio.sleep(0)

    async def run(self):
        """"
            Monitor the data stream and process each tick.
        """
        async for tickers in self.ib.pendingTickersEvent:
            for ticker in tickers:
                _tick = self.sym_ticks[ticker.contract.symbol]
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
