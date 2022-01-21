#  let's use this file to compose the network, and then set it running.
#
# also, need a way to replay ticks if we're going to test this at night.
# Can read and process our saved file with a little work.
# don't forget to add in logging real soon now.
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from streamer import connect, ticks, candles, emacalc

comment_2021_12_29 = """
I've thought about this some, and decided to accept the inefficiency of filtering 
ticks for all subscriptions, since it makes the long lived tasks simpler and avoids
queues and other heavy items that might be even worse.  Will continue forward with this
for now...
"""
comment_2022_01_05 = """
decided I was right(er) at the beginning.  Will instantiate streaming objects
then explicitly describe network on a per security basis.
"""


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


if __name__ == "__main__":
    connection = connect.Connection()
    connection.select(connect.Connection.btcjo)
    ib = connection.connect()
    asyncio.run(create(ib), debug=False)
