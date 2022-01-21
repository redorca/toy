# std lib
import asyncio
import datetime
import statistics
import sys
from typing import Optional, Union

# PyPI


import ib_insync

# local

# in case we use Ticker ticktypes:
# 1 highest bid
# 2 lowest offer
# 4 last price traded
# 5 last size
# 6 daily high
# 7 daily low
# 8 volume for the day / 100
# 9 previous day's close
# 14 this day's open
# 49 trading halted.
# size(?) vals: -1=unknown, 0=not halted, 1=general halt, 2=volatility halt


class Candle:
    def __init__(
        self,
        symbol: str,
        opening: float,
        high: float,
        low: float,
        closing: float,
        start: datetime.datetime,
        end: datetime.datetime,
        average: float,
        median: float,
        number_ticks: int,
    ):
        self.symbol = symbol
        self.opening = opening
        self.high = high
        self.low = low
        self.closing = closing
        self.start = start
        self.end = end
        self.average = average
        self.median = median
        self.number_ticks = number_ticks

    def __str__(self):
        val = (
            f"Candle {self.symbol}: open:{self.opening}"
            f", high:{self.high}, low:{self.low}"
            f", close:{self.closing}, num_ticks:{self.number_ticks}"
        )
        return val


class CandleMakerBase:
    def __init__(self):
        self.high = -1
        self.low = sys.maxsize
        # self.max_ticks = 1000
        self.volume = 0
        self.start_time = None
        self.start_timestamp = None
        self.end_time = None
        self.prices = list()
        self.timestamps = list()
        self.symbol: Optional[str] = None

    def re_init(self):
        self.high = -1
        self.low = sys.maxsize
        self.volume = 0
        self.start_time = None
        self.start_timestamp = None
        self.end_time = None
        # https://stackoverflow.com/questions/29193127/is-it-faster-to-truncate-a-list-by-making-it-equal-to-a-slice-or-by-using-del
        del self.prices[:]
        del self.timestamps[:]

    def run_a(self, ticker: ib_insync.Ticker):
        # TODO: need a check here. when market is closed, we continue to get ticks
        # of the last trade, even though halted==0.0. could look for weird
        # data: bid & ask=-1, bidSize&askSize=0, volume does not increase.
        if self.start_time is None:  # cleared in re_init
            self.start_time = ticker.time
            self.start_timestamp = ticker.time.timestamp()
            if self.symbol is None:
                self.symbol = ticker.contract.symbol
        self.high = max((self.high, ticker.last))
        self.low = min((self.low, ticker.last))
        self.volume += ticker.lastSize
        self.end_time = ticker.time
        self.prices.append(ticker.last)  # most recent price
        self.timestamps.append(ticker.time.timestamp())  # a float, not a datetime


class CandleMakerTimed(CandleMakerBase):
    """signature to change to take a list of Calculators to send Candle to"""

    def __init__(self, seconds: Union[int, float] = 60):
        super().__init__()
        self.seconds = seconds

    def run_a(self, ticker: ib_insync.Ticker):
        # https://stackoverflow.com/questions/29193127/is-it-faster-to-truncate-a-list-by-making-it-equal-to-a-slice-or-by-using-del
        timestamp = ticker.time.timestamp()
        if timestamp - self.start_timestamp > self.seconds:  # may exceed seconds
            # time to emit a candle
            candle = Candle(
                symbol=ticker.contract.symbol,
                opening=self.prices[0],
                high=max(self.prices),
                low=min(self.prices),
                closing=self.prices[-1],
                start=self.start_time,
                end=self.end_time,
                average=statistics.mean(self.prices),
                median=statistics.median(self.prices),
                number_ticks=len(self.prices),
            )
            self.re_init()
            return candle

        # to keep timed candles from running late, we check time first and
        # emit existing candle if incoming time is over limit
        super().run_a(ticker)
        return None


class CandleMakerCounted(CandleMakerBase):
    def __init__(self, number: int = 100):
        super().__init__()
        self.number_of_ticks = number

    async def run_a(self, ticker: ib_insync.Ticker):
        super().run_a(ticker)

        if len(self.prices) >= self.number_of_ticks:
            # emit a candle
            candle = Candle(
                symbol=ticker.contract.symbol,
                opening=self.prices[0],
                high=max(self.prices),
                low=min(self.prices),
                closing=self.prices[-1],
                start=self.start_time,
                end=self.end_time,
                average=statistics.mean(self.prices),
                median=statistics.median(self.prices),
                number_ticks=len(self.prices),
            )
            self.re_init()
            return candle
        else:
            return None


class CandleMakerDollarVolume(CandleMakerBase):
    def __init__(self, dollar_volume=1000000):
        super().__init__()
        # our candles contain >= max_dollar_volume
        self.max_dollar_volume: int = dollar_volume
        self.dollar_volume: float = 0.0

    def re_init(self):
        self.dollar_volume = 0
        super().re_init()

    async def run_a(self, ticker: ib_insync.Ticker):
        super().run_a(ticker)
        self.dollar_volume += float(ticker.last) * float(ticker.lastSize)
        if self.dollar_volume >= self.max_dollar_volume:
            # emit a candle
            candle = Candle(
                symbol=ticker.contract.symbol,
                opening=self.prices[0],
                high=max(self.prices),
                low=min(self.prices),
                closing=self.prices[-1],
                start=self.start_time,
                end=self.end_time,
                average=statistics.mean(self.prices),
                median=statistics.median(self.prices),
                number_ticks=len(self.prices),
            )
            self.re_init()
            return candle
        else:
            return None


if __name__ == "__main__":
    from ib_insync import Stock, Ticker, TickData
    import datetime

    t = ib_insync.Ticker(
        contract=Stock(symbol="AAPL", exchange="SMART", currency="USD"),
        time=datetime.datetime(
            2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
        ),
        bid=-1.0,
        bidSize=0.0,
        ask=-1.0,
        askSize=0.0,
        last=172.12,
        lastSize=1.0,
        volume=1504361.0,
        open=179.15,
        high=181.14,
        low=170.75,
        close=179.3,
        ticks=[
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=1,
                price=-1.0,
                size=0.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=2,
                price=-1.0,
                size=0.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=4,
                price=172.12,
                size=1.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=5,
                price=172.12,
                size=1.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=8,
                price=-1.0,
                size=1504361.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=6,
                price=181.14,
                size=0.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=7,
                price=170.75,
                size=0.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=9,
                price=179.3,
                size=0.0,
            ),
            TickData(
                time=datetime.datetime(
                    2021, 12, 17, 2, 55, 26, 675099, tzinfo=datetime.timezone.utc
                ),
                tickType=14,
                price=179.15,
                size=0.0,
            ),
        ],
    )

    cbc = CandleMakerCounted(number=5)
    cbt = CandleMakerTimed(seconds=30)

    for _ in range(10):
        asyncio.run(cbc.run_a(t), debug=True)
