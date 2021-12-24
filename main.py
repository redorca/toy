#  lets use this file to compose the network, and then set it running.
#
# also, need a way to replay ticks if we're going to test this at night.
# Can read and process our saved file with a little work.
# don't forget to add in logging real soon now.
import asyncio
import ib_insync
from streamer import conf, connect, ticks, candles, emacalc


async def run_a(
    tick_src: ticks.Ticks,
    candle_maker: candles.CandleMakerBase,
    ema_calculator: emacalc.EmaCalculator,
):
    while True:
        tick = await tick_src.run_a()  # should keep emitting ticks
        candle = await candle_maker.run_a(tick)  # will filter them down to candles
        if candle is None:
            continue
        print(candle)
        ema = await ema_calculator.run_a(candle)  # incomplete
        if ema is None:
            continue
        # print(ema)

comment_2021_12_23 = """
this version of main and run_a don't work well.  It is built with the expectation
that each tick_src only sees the ticker symbols it registers for.  This would have
been convenient, in that ticks would be one stock, and could feed one candlemaker.
But, when we instantiate two tick_src's, each get the sum of the subscriptions. 
Current implementation filters out incoming that don't match this objects subscription,
but that means that multiple tick src's will be mostly filtering out redundant data.
BUT, if we have one tick src, we can't separate the streams until the ticks come
in, and that makes it {hard|difficult to understand how to} have separate tasks per
data flow on a single stock.  I was counting on each object in the flow to have to
maintain state (candle, ema, etc) for 1 data flow, that is one stock.

Anyway, this needs a refactoring to work better. 

Current thought is a single tick src with all symbols, then read the symbol and
use a case statement to feed to another task which is just the single stock flow.
This would mean main would need to figure out how to transmit ticks to long lived tasks.
This implies queues, which I'm trying to avoid.

Alternate implementation would be for tick src to have a list of places to forward 
different stock ticks to.  This embeds the network into the objects, which I was trying
to avoid.  
"""
async def main(connection_info: dict):
    # this is where we compose the network (Arun's block diagram)
    # first make objects

    conn = connect.Connection()
    conn.select(connection_info)
    ib = await conn.connect_async()

    task_set = set()
    for symbol in ("AAPL", "TSLA"):  #
        # connection_info["symbol"] = symbol
        # make the processing objects
        tick_src = ticks.Ticks(ib, symbol)
        candle_maker = candles.CandleMakerDollarVolume(dollar_volume=100000)
        ema_calculator = emacalc.EmaCalculator()

        task_set.add(
            asyncio.create_task(
                run_a(tick_src, candle_maker, ema_calculator), name=symbol
            )
        )
    # print('in main')
    results = await asyncio.gather(*task_set)
    return results


if __name__ == "__main__":
    import sys

    connection_info = connect.Connection.btcjo
    asyncio.run(main(connection_info=connection_info), debug=False)
