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
    """Ticks objects clean up the problems with incoming tickers,
    and deliver a "clean" stream of Ticker objects.
    It removes redundant ticker data,
    and fills in when volume increases without showing a matching lastSize"""

    def __init__(self, ib_conn, symbol: str):
        self.ib = ib_conn
        self.symbol = symbol
        self.queued_tickers = deque(maxlen=32)  # not dequeue, but double ended queue
        self.latest_volume = -1
        self.post_queue_latest_volume = -1

    async def run_a(self):
        logger.debug(f"starting {__name__}.run_a")
        contract = ibi.Stock(self.symbol, "SMART", "USD")
        tkr = self.ib.reqMktData(contract, snapshot=False)
        # logger.debug(f"type of tkr is {type(tkr)}")
        async for tickers in self.ib.pendingTickersEvent:
            logger.debug(tickers)
            for ticker in tickers:
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
                # can only return once per call, so we can get backed up
                # use "bad" ticker events to help drain the queue
                if len(self.queued_tickers) > 0:
                    ticker_ = self.queued_tickers.popleft()
                    await asyncio.sleep(0)  # printing and scrolling is slow
                    logger.debug(
                        f"{self.symbol}"
                        #     f":${ticker.last}"
                        #     f" sz:{ticker.lastSize}"
                        #     f" vol:{ticker.volume}"
                        f"  {len(self.queued_tickers)} tickers in queue"
                    )
                    # self.post_queue_latest_volume = ticker_.volume
                    return ticker_
                else:
                    return None

    async def cycle(self):
        print("in cycle")
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
