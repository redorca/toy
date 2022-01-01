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


comment_2021_12_29 = """
I've thought about this some, and decided to accept the inefficiency of filtering 
ticks for all subscriptions, since it makes the long lived tasks simpler and avoids
queues and other heavy items that might be even worse.  Will continue forward with this
for now...
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
