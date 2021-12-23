# std lib
import asyncio
import logging

# PyPI
import ib_insync as ibi

# local
from streamer import conf


class Ticks:
    clientid_var = 10

    def __init__(self, ib_conn, symbol: str):
        self.ib = ib_conn
        self.symbol = symbol

    async def run_a(self):
        contract = ibi.Stock(self.symbol, "SMART", "USD")
        self.ib.reqMktData(contract)
        async for tickers in self.ib.pendingTickersEvent:
            for ticker in tickers:
                # here's where we feed tickers to CandleBuilder(symbol)
                print(ticker)
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
