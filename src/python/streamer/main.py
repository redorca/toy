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
import argparse as args
import configparser as cfg

# PyPI
# import uvloop

# local modules
import time

import transcend
from streamer import connect, ticks, candles, emacalc
from streamer import davelogging as dl

logger = dl.logger(__name__, dl.DEBUG, dl.logformat)
# logger.debug(f"__name__ is {__name__}")  # package name: streamer.davelogging
# logger.debug(f"__file__ is {__file__}")  # file: /whole/path/to/davelogging.py


async def create(ib, Symbols, bundler):
    # this is where we add processing block (Arun's block diagram)
    # first make objects

    await bundler.register(Symbols)
    task = asyncio.create_task(compose(bundler, ib))
    results = None
    if not task is None:
        results = await asyncio.gather(task)
    if results is None: logger.debug("No task created.")
    return results


async def compose(Bundle, ibi):
    duplicate = 0
    skipped = 0
    ema_calculator = emacalc.EmaCalculator()
    while True:
        """
                Run a loop for each stock/security a Tick() object represents:
        """
        tkr = await Bundle.run()
        if tkr is None:
            skipped += 1
            continue

        localTick = await Bundle.tick_for_ticker(tkr)

        ############################################################
        # quitting for the night, but
        # this needs to migrate into candles that ignore the lastSize
        # and instead follow the volume indicator
        # discovered that redundant ticks of small size are emitted,
        # and there are missing ticks where the volume jumps.
        # need to decide how to handle prices.
        # average of earlier and later tick?
        #
        if localTick.last_price == tkr.last and localTick.last_volume == tkr.volume:
            duplicate += 1
            continue

        logger.debug(
            f"{tkr.rtTime}"
            f" {tkr.contract.symbol}"
            f" ${tkr.last:0.2f}"
            f" sz:{tkr.lastSize}"
            f" vol:{tkr.volume}"
        )
        if localTick.last_price == tkr.last and localTick.last_volume == tkr.volume:
            continue

        localTick.largest_size = max(localTick.largest_size, tkr.lastSize)
        if localTick.volume_initialized     \
           and (tkr.volume - tkr.lastSize - localTick.last_volume > 10.0):
            logger.error(
                "========== BIG JUMP ==============> "
                f"{tkr.contract.symbol}"
                f" new vol: {tkr.volume} != sum: {tkr.lastSize + localTick.last_volume}"
                f" difference: {tkr.volume - localTick.last_volume - tkr.lastSize}"
            )

        localTick.last_price  = tkr.last
        localTick.last_volume = tkr.volume
        localTick.volume_initialized = True

        candle_maker = await Bundle.candle_for_tick(localTick)
        candle = await candle_maker.run_a(tkr)
        if candle is None:
            continue
        logger.info(f"=======  CANDLE  =========> {candle}")
        ema = await ema_calculator.run_a(candle)  # incomplete
        if ema is None:
            continue


async def main(gateway, secType, ticksFile):
    try:
        connection = connect.Connection(gateway)
        start = time.perf_counter()
        ib = await connection.connect_async()
        logger.debug(f"connection took {time.perf_counter() - start} seconds")
    except (TimeoutError,
            asyncio.exceptions.TimeoutError,
            ConnectionError) as err:
        print(f"Exception: {err}")
        return

    kfg = cfg.ConfigParser()
    kfg.read(ticksFile)

    if 'options' in kfg.sections():
        sec_type = 'OPT'
        section = 'options'
    elif 'stocks' in kfg.sections():
        section = 'stocks'
        sec_type = 'STK'
    else:
        logger.debug(f'unable to determine whether this set of securities are stocks or options')
        return

    logger.debug(f'Security type: {sec_type}')
    funds = transcend.bundles.Bundle(ib, list(kfg[section][sec_type].split("\n"))[1:], secType=sec_type)
    await create(ib, funds.list(), funds)

HelP = dict()
HelP['stocks'] = "Monitor stocks from the IB Gateway."
HelP['options'] = "Monitor options from the IB Gateway."
HelP['file'] = "Contains the list of securities to track."

if __name__ == "__main__":
    cmdParse = args.ArgumentParser('Roon')
    cmdParse.add_argument('-S', '--stocks', help=HelP['stocks'], action='store_true', default=False)
    cmdParse.add_argument('-O', '--options', help=HelP['options'], action='store_true', default=False)
    cmdParse.add_argument('-F', '--file', help=HelP['file'], nargs=1, required=True, action='store')

    cmdLine = cmdParse.parse_args()
    if not cmdLine.stocks and not cmdLine.options:
        print(f'One of --stock or --options must be specified.')
        exit()

    # One of stocks, options must be true.
    secType = 'OPT'
    if cmdLine.stocks:
        secType = 'STK'

    gateway = connect.Btchfpaper()
    ticksFile = cmdLine.file
    asyncio.run(main(gateway, secType, ticksFile), debug=False)
