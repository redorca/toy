# std lib
import asyncio
from statistics import mean, median
import sys
import time

# PyPI
import time

import ib_insync as ib

# local


class CandleBuilder:
    def __init__(self):
        self.high = -1
        self.low = sys.maxsize
        self.max_ticks = 1000
        self.volume = 0
        self.start_time = time.monotonic_ns()
        self.prices = list()

    async def build(self, ticker: ib.Ticker):
        stock = ticker.contract
        symbol = stock.symbol
        timestamp = ticker.time.timestamp()  # conversions, probably
        last = ticker.last
        self.prices.append(last)
        # https://stackoverflow.com/questions/29193127/is-it-faster-to-truncate-a-list-by-making-it-equal-to-a-slice-or-by-using-del
        need_to_delete = len(self.prices) - 100
        if need_to_delete > 0:
            del self.prices[0:need_to_delete]
        average = mean(self.prices)
        median_ = median(self.prices)
        problem = """
        doesn't calling through on every nth cause problems with the return stack?
        -or- maybe not.  Every 100th time you call build, something else happens...
        """
