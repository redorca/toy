# system modules
import asyncio
import aiofiles
from random import randint
import sys

# PyPI
import ib_insync as ibi

# local
import conf


class Ticks:
    clientid_var = 10

    def __init__(
        self,
        host: str = conf.HOST,
        port: int = conf.PORT,
        account: str = conf.ACCOUNT,
        timeout: int = 10,
        clientid: int = clientid_var,
        symbol: str = "AAPL",
    ):
        self.host = host
        self.port = port
        self.account = account
        self.timeout = timeout
        self.clientid = clientid
        Ticks.clientid_var += 1
        self.symbol = symbol
        self.ib = None
        self.candlebuilder = None  # actually CandleBuilder() one object per symbol

    async def run(self):
        self.ib = ibi.IB()
        with await self.ib.connectAsync(
            host=self.host,
            port=self.port,
            clientId=self.clientid,
            account=self.account,
            timeout=self.timeout,
        ):
            async with aiofiles.open("short_saved_tickers.txt", "w") as f:
                contract = ibi.Stock(self.symbol, "SMART", "USD")
                self.ib.reqMktData(contract)
                counter = 1
                async for tickers in self.ib.pendingTickersEvent:
                    for ticker in tickers:
                        # here's where we feed tickers to CandleBuilder(symbol)
                        print(counter, ticker)
                        ts = ticker.__str__() + "\n"
                        await f.write(ts)
                        if counter >= 10:
                            await f.close()
                            sys.exit(0)
                        counter += 1

    def stop(self):
        self.ib.disconnect()


connection_info = {
    "host": conf.HOST,
    "port": conf.PORT,
    "account": conf.ACCOUNT,
    "timeout": 5,
}
app = Ticks(**connection_info)

try:
    asyncio.run(app.run())
except (KeyboardInterrupt, SystemExit):
    app.stop()
