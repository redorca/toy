# std lib
import asyncio
import time
from collections import deque
from math import isclose


# PyPI
import ib_insync as ibi
import numpy as np

# local
from streamer import connect
from streamer import davelogging as dl


logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
# logger.debug(f"__name__ is {__name__}")  # package name: streamer.davelogging
# logger.debug(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py


class Ticks:
    """
        Ticks objects clean up the problems with incoming tickers,
        and deliver a "clean" stream of Ticker objects.
        It removes redundant ticker data,
        and fills in when volume increases without showing a matching lastSize
    """

    def __init__(self, ib_conn, symbol: str):
        self.last_volume = 0
        self.volume_initialized = True
        self.largest_size = 0
        self.ib = ib_conn
        self.symbol = symbol
        self.queued_tickers = deque(maxlen=32)  # not dequeue, but double ended queue
        self.latest_volume = -1

    async def run_a(self):
        # logger.debug(f"starting {__name__}.run_a")
        """
            Iniitiate contact with IB and establish a data stream for a particular subscription.
        """
        contract = ibi.Stock(self.symbol, "SMART", "USD")

        # start the ticker stream and events. The variable, tkr,  is a throw away here.
        tkr = self.ib.reqMktData(contract, snapshot=False)
        # logger.debug(f"type of tkr is {type(tkr)}")
        # logger.debug(f"run_a for {self.symbol}")
        async for tickers in self.ib.pendingTickersEvent:
            for ticker in tickers:
                # logger.debug(f"{self.symbol} ticker: {ticker.contract.symbol}")
                await asyncio.sleep(0)
                # logger.debug(ticker)
                # each Ticks object will see all subscriptions
                # first check for redundant ticks
                if (  # valid ticker data checks
                    ticker.contract.symbol == self.symbol
                    and ticker.volume > self.latest_volume
                    and not np.isnan(ticker.volume)
                    and ticker.bidSize > 0
                    # and ticker.askSize > 0  # apparently redundant to bidSize
                    and not np.isnan(ticker.halted)
                    and ticker.halted < 0.5
                ):
                    self.latest_volume = ticker.volume
                    self.queued_tickers.append(ticker)
                    q_len = len(self.queued_tickers)
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
                if len(self.queued_tickers) > 0:
                    """
                        If this particular ticker stream (subscription) actually contains
                        ticks then pop the oldest from the queue and return it.
                    """
                    ticker_ = self.queued_tickers.popleft()
                    await asyncio.sleep(0)  # printing and scrolling is slow
                    # logger.debug(
                    #     f"{self.symbol}"
                    #     #     f":${ticker.last}"
                    #     #     f" sz:{ticker.lastSize}"
                    #     #     f" vol:{ticker.volume}"
                    #     f"  {len(self.queued_tickers)} tickers in queue"
                    # )
                    return ticker_
                else:
                    """
                        The queue hasn't any elements so return None.
                        Async calls always return some value else async gather won't finish.
                    """
                    return None

    async def run_b(self):
        logger.debug(f"starting {__name__}.run_b")
        """
            Run only the pendingTickersEvent monitor
        """
        async for tickers in self.ib.pendingTickersEvent:
            logger.debug(f"tickers:")
            for ticker in tickers:
                logger.debug(f"{self.symbol} ticker: {ticker.contract.symbol}")
                await asyncio.sleep(0)
                # logger.debug(ticker)
                # each Ticks object will see all subscriptions
                # first check for redundant ticks
                if (  # valid ticker data checks
                    ticker.volume > self.latest_volume
                    and not np.isnan(ticker.volume)
                    and ticker.bidSize > 0
                    # and ticker.askSize > 0  # apparently redundant to bidSize
                    and not np.isnan(ticker.halted)
                    and ticker.halted < 0.5
                ):
                    self.latest_volume = ticker.volume
                    self.queued_tickers.append(ticker)
                    q_len = len(self.queued_tickers)
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
                if len(self.queued_tickers) > 0:
                    """
                        If this particular ticker stream (subscription) actually contains
                        ticks then pop the oldest from the queue and return it.
                    """
                    ticker_ = self.queued_tickers.popleft()
                    await asyncio.sleep(0)  # printing and scrolling is slow
                    # logger.debug(
                    #     f"{self.symbol}"
                    #     #     f":${ticker.last}"
                    #     #     f" sz:{ticker.lastSize}"
                    #     #     f" vol:{ticker.volume}"
                    #     f"  {len(self.queued_tickers)} tickers in queue"
                    # )
                    # return ticker_
                    await self.genData(ticker_)
                else:
                    """
                        The queue hasn't any elements so return None.
                        Async calls always return some value else async gather won't finish.
                    """
                    # return None
                    logger.debug(f"queue is 0")

    async def genData(self, tick): 
        ############################################################
        # quitting for the night, but
        # this needs to migrate into candles that ignore the lastSize
        # and instead follow the volume indicator
        # discovered that redundant ticks of small size are emitted,
        # and there are missing ticks where the volume jumps.
        # need to decide how to handle prices.
        # average of earlier and later tick?
        #
        logger.debug(
            f"{tick.contract.symbol}"
            f" ${tick.last:0.2f}"
            f" sz:{tick.lastSize}"
            f" vol:{tick.volume}"
        )
        self.largest_size = max(self.largest_size, tick.lastSize)
        # logger.debug(f"largest transaction size seen so far: {largest_size}")
        if self.volume_initialized and (tick.volume - tick.lastSize - self.last_volume > 10.0):
            logger.error(
                "========== BIG JUMP ==============> "
                f"{tick.contract.symbol}"
                f" new vol: {tick.volume} != sum: {tick.lastSize + self.last_volume}"
                f" difference: {tick.volume - self.last_volume - tick.lastSize}"
            )
        self.last_volume = tick.volume
        self.volume_initialized = True
        candle = await candle_maker.run_a(tick)  # will filter them down to candles
        logger.info(f"=======  CANDLE  =========> {candle}")
        ema = await ema_calculator.run_a(candle)  # incomplete


    async def cycle(self):
        """
            create an event loop on run_a()
        """
        logger.debug(f"In Cycle:")
        while True:
            something = await self.run_a()
            if something is not None:
                print(something)

    def stop(self):
        self.ib.disconnect()


if __name__ == "__main__":

    gateway = connect.Btcjopaper()
    conn = connect.Connection(gateway)
    start = time.perf_counter()
    ib = conn.connect()
    print(f"connection took {time.perf_counter()-start:4.2f} seconds")
    app = Ticks(ib, "TSLA")

    try:
        """
            Start the asyncio event loop
        """
        asyncio.run(app.cycle())
    except (KeyboardInterrupt, SystemExit):
        app.stop()

# Notes on ticks
# During market hours we see redundant, small transactions.
# That is, the size is > 0, and the price is the same, but the volume does not increase.
# Also, on 2022-01-25, there were several (many) jumps in TSLA volumes
# that were not visible in transaction size.
# (Maybe that's data from some other exchange where they get a summary?)
# Decided to drop redundant ticks, but leave it to candles or later processing
# to decide what to do.  Recommendation now would be to look at the volume increases,
# which are now guaranteed to happen, and figure out what to do about estimating
# price.
