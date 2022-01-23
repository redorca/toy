#  let's use this file to compose the network, and then set it running.
#
# also, need a way to replay ticks if we're going to test this at night.
# Can read and process our saved file with a little work.
# don't forget to add in logging real soon now.

# standard library
import asyncio

# PyPI
# import uvloop

# local modules
import time

from streamer import connect, ticks, candles, emacalc
from streamer import davelogging as dl

logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
logger.debug(f"__name__ is {__name__}")  # package name: streamer.davelogging
logger.debug(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py


async def create(ib):
    # this is where we add processing block (Arun's block diagram)
    # first make objects

    task_set = set()
    for symbol in ("TSLA",):  # "AAPL",
        # connection_info["symbol"] = symbol
        # make the processing objects
        tick_src = ticks.Ticks(ib, symbol)
        candle_maker = candles.CandleMakerDollarVolume(dollar_volume=100000)
        ema_calculator = emacalc.EmaCalculator()

        task_set.add(
            asyncio.create_task(
                compose(tick_src, candle_maker, ema_calculator), name=symbol
            )
        )
    # print('in main')
    results = await asyncio.gather(*task_set)
    return results


async def compose(
    tick_src: ticks.Ticks,
    candle_maker: candles.CandleMakerBase,
    ema_calculator: emacalc.EmaCalculator,
):
    """this connects the blocks from create"""
    while True:
        tick = await tick_src.run_a()  # should keep emitting ticks
        print(tick)
        candle = await candle_maker.run_a(tick)  # will filter them down to candles
        if candle is None:
            continue
        print(candle)
        ema = await ema_calculator.run_a(candle)  # incomplete
        if ema is None:
            continue
        # print(ema)


async def main(gateway):
    connection = connect.Connection(gateway)
    start = time.perf_counter()
    ib = await connection.connect_async()
    print(f"connection took {time.perf_counter() - start} seconds")
    # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    await create(ib)


if __name__ == "__main__":
    gateway = connect.Btchfpaper()
    asyncio.run(main(gateway), debug=True)
