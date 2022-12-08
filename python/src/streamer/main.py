#  let's use this file to compose the network, and then set it running.
#
# also, need a way to replay ticks if we're going to test this at night.
# Can read and process our saved file with a little work.
# don't forget to add in logging real soon now.

"""
    TODO: Add flow monitors to allow flow rate adjustments to the data
          streams, track stream efficiency (meaningful ticks / all ticks).

    TODO: Add wall clock check to know whether data is from an active market or not.
"""

# standard library
import asyncio
from collections import defaultdict
from math import isclose

# PyPI
# import uvloop

# local modules
import time

from streamer import connect, ticks, candles, emacalc
from streamer import davelogging as dl

logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
# logger.debug(f"__name__ is {__name__}")  # package name: streamer.davelogging
# logger.debug(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py


async def create(ib):
    # this is where we add processing block (Arun's block diagram)
    # first make objects

    task_set = set()
    dollar_volume = defaultdict(lambda: 100000, {"RSP": 50000})
    for symbol in (
        "TSLA",
        "AAPL",
        "RSP",
    ):  #
        # connection_info["symbol"] = symbol
        # make the processing objects
        tick_src = ticks.Ticks(ib, symbol)
        candle_maker = candles.CandleMakerDollarVolume(dollar_volume[symbol])
        ema_calculator = emacalc.EmaCalculator()

        """
            Create a task per Tick() security, add it to a set for later obsrvation
        """
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
    last_volume = 0
    volume_initialized = False
    largest_size = 0
    while True:
        """
                Run a loop for each stock/security a Tick() object represents:
        """
        tick = await tick_src.run_a()  # should keep emitting ticks
        if tick is None:
            continue

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
        largest_size = max(largest_size, tick.lastSize)
        # logger.debug(f"largest transaction size seen so far: {largest_size}")
        if volume_initialized and (tick.volume - tick.lastSize - last_volume > 10.0):
            logger.error(
                "========== BIG JUMP ==============> "
                f"{tick.contract.symbol}"
                f" new vol: {tick.volume} != sum: {tick.lastSize + last_volume}"
                f" difference: {tick.volume - last_volume - tick.lastSize}"
            )
        last_volume = tick.volume
        volume_initialized = True
        candle = await candle_maker.run_a(tick)  # will filter them down to candles
        if candle is None:
            continue
        logger.info(f"==========  CANDLE  ==============> {candle}")
        ema = await ema_calculator.run_a(candle)  # incomplete
        if ema is None:
            continue
        # print(ema)


async def main(gateway):
    connection = connect.Connection(gateway)
    start = time.perf_counter()
    ib = await connection.connect_async()
    logger.debug(f"connection took {time.perf_counter() - start} seconds")
    # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    await create(ib)


if __name__ == "__main__":
    gateway = connect.Btchfpaper()
    asyncio.run(main(gateway), debug=False)
