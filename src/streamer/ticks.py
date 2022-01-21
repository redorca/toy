# std lib
import asyncio
import logging
import sys
from math import isclose

import davelogging as dl

# PyPI
import ib_insync as ibi

# local
from streamer import config, connect

# print(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py
# print(f"__name__ is {__name__}")  # package name: streamer.davelogging
logger = dl.logger(__name__, dl.DEBUG, dl.logformat)


class Ticks:
    clientid_var = 10

    def __init__(self, ib_conn, symbol: str):
        self.ib = ib_conn
        self.symbol = symbol
        self.queued_tickers = list()
        self.latest_volume = -1

    async def run_a(self):
        contract = ibi.Stock(self.symbol, "SMART", "USD")
        self.ib.reqMktData(contract)
        async for tickers in self.ib.pendingTickersEvent:
            for ticker in tickers:
                if (  # checking for a redundant tick:
                    # each Ticks object will see all subscriptions
                    self.symbol != ticker.contract.symbol
                    or self.latest_volume == ticker.volume
                    # after market close, we get -$1 ticks of size 0.0
                    or ticker.bidSize < 1.1  # don't use == with floats
                    or ticker.bid < 0.0
                    or ticker.ask < 0.0
                    or isclose(ticker.halted, 0.0, abs_tol=0.1)
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

    def stop(self):
        self.ib.disconnect()


if __name__ == "__main__":

    connection_info = {
        "host": config.HOST,
        "port": config.PORT,
        "account": config.ACCOUNT,
        "timeout": 5,
    }
    ib_conn = connect.Connection()
    ib_conn.select(ib_conn.btcjo)
    ib = ib_conn.connect()
    app = Ticks(ib, "IBM")

    try:
        asyncio.run(app.run_a())
    except (KeyboardInterrupt, SystemExit):
        app.stop()
