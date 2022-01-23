# std lib
import asyncio
import time
from math import isclose


# PyPI
import ib_insync as ibi

# local
from streamer import connect
from streamer import davelogging as dl


logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
logger.debug(f"__name__ is {__name__}")  # package name: streamer.davelogging
logger.debug(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py


class Ticks:
    def __init__(self, ib_conn, symbol: str):
        self.ib = ib_conn
        self.symbol = symbol
        self.queued_tickers = list()
        self.latest_volume = -1

    async def run_a(self):
        logger.debug("in run_a")
        contract = ibi.Stock(self.symbol, "SMART", "USD")
        tkr = self.ib.reqMktData(contract, snapshot=False)
        logger.debug(f"type of tkr is {type(tkr)}")
        async for tickers in self.ib.pendingTickersEvent:
            logger.debug(tickers)
            for ticker in tickers:
                # checking for a redundant tick
                if (
                    # each Ticks object will see all subscriptions
                    self.symbol
                    != ticker.contract.symbol
                    # or self.latest_volume == ticker.volume
                    # # after market close, we get -$1 ticks of size 0.0
                    # or ticker.bidSize < 1.1  # don't use == with floats
                    # or ticker.bid < 0.0
                    # or ticker.ask < 0.0
                    # or isclose(ticker.halted, 0.0, abs_tol=0.1)
                ):  # redundant or bad ticker, ignore.
                    if len(self.queued_tickers) == 0:
                        continue
                    else:  # use redundant ticker as opportunity to reduce queue
                        ticker = self.queued_tickers.pop(0)
                        logger.debug(
                            f"{self.symbol}"
                            f":{ticker.last}"
                            f":{ticker.volume}"
                            f":{ticker.lastSize}"
                            f"  dequeued len={len(self.queued_tickers)}"
                        )
                        return ticker
                else:  # store qualified tickers in a queue
                    self.latest_volume = ticker.volume
                    self.queued_tickers.append(ticker)
                    logger.debug(
                        f"{self.symbol}"
                        f":{self.queued_tickers[-1].last}"
                        f":{self.queued_tickers[-1].volume}"
                        f":{self.queued_tickers[-1].lastSize}"
                        f"  enqueued len={len(self.queued_tickers)}"
                    )

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
