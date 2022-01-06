# std lib
import asyncio
import logging
import sys
from math import isclose

import davelogging as dl

# PyPI
import ib_insync as ibi

# local
from streamer import conf

# print(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py
# print(f"__name__ is {__name__}")  # package name: streamer.davelogging
logger = dl.logger(__name__, dl.DEBUG, dl.logformat)


class Ticks:
    clientid_var = 10

    def __init__(self, ib_conn, symbol: str):
        self.ib = ib_conn
        self.symbol = symbol
        self.queued_tickers = list()

    async def run_a(self):
        contract = ibi.Stock(self.symbol, "SMART", "USD")
        self.ib.reqMktData(contract)
        async for tickers in self.ib.pendingTickersEvent:
            for ticker in tickers:
                # each Ticks object will see all subscriptions
                if self.symbol != ticker.contract.symbol:
                    continue
                # after market close, we get -$1 ticks of size 0.0
                if (
                    ticker.bidSize < 0.0001  # don't use == with floats
                    or ticker.bid < 0.0
                    or ticker.ask < 0.0
                ):
                    continue
                # store qualified tickers in a queue
                self.queued_tickers.append(ticker)
                logger.debug(
                    f"{self.symbol}:{self.queued_tickers[-1].contract.symbol}:"
                    f"{self.queued_tickers[-1].bidSize} "
                    f"queue len={len(self.queued_tickers)}"
                )

            if len(self.queued_tickers) > 0:
                ticker = self.queued_tickers.pop(0)
                # logger.debug(f"{self.symbol}:{ticker.contract.symbol}:{ticker.bidSize}")
                # only return 1 tick, but this logic will get exercised with every
                # (wrong) incoming tick, too.
                return ticker

    def stop(self):
        self.ib.disconnect()


if __name__ == "__main__":

    connection_info = {
        "host": conf.HOST,
        "port": conf.PORT,
        "account": conf.ACCOUNT,
        "timeout": 5,
    }
    app = Ticks(**connection_info)

    try:
        asyncio.run(app.run_a())
    except (KeyboardInterrupt, SystemExit):
        app.stop()
