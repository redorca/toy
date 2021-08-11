import asyncio
import time
import ib_insync as ibs

import nest_asyncio

nest_asyncio._patch_asyncio()

import bot.conf

in_queue = asyncio.Queue(1000)
out_queue = asyncio.Queue(1000)


def options(ib):
    option_chains = []
    for position in ib.positions(bot.conf.ACCOUNT):
        if isinstance(position.contract, ibs.Stock):
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            foo = ib.reqSecDefOptParams(symbol, '', sec_type, conId)
            option_chains.append(foo)
    return option_chains


async def options_async(ib):
    option_chains = []
    cnt = 0
     # while True:
    #     # await asyncio.sleep(5)
    #     await in_queue.put(f"hi dave {cnt}")
    #     cnt+= 1
    for position in ib.positions(bot.conf.ACCOUNT):
        start = time.perf_counter()
        if isinstance(position.contract, ibs.Stock):
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            option_chain = await ib.reqSecDefOptParamsAsync(symbol, '', sec_type, conId)
            await in_queue.put(option_chain)
            # option_chains.append(option_chain)
            # print("inner loop time {}".format(time.perf_counter()-start))


async def read_task():
    start = time.perf_counter()
    while True:
        print("waiting for data...")
        option_chain_print = await in_queue.get()
        in_queue.task_done()
        end = time.perf_counter()
        print("that took {} secs".format(end - start), end="//")
        print(option_chain_print,in_queue.qsize())
        # await asyncio.sleep(2)

async def awaiting_for_complete(event1, event2):
    await asyncio.gather(event1, event2)


async def main(ib):
    loop = asyncio.get_event_loop()
    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    start = time.perf_counter()
    if True:
        # loop1 = asyncio.new_event_loop()
        # loop2 = asyncio.new_event_loop()
        option_chains_read = loop.create_task(read_task())
        # create a queue of data.
        option_chains_options = loop.create_task(options_async(ib))
        # option_chains = asyncio.run(options_async(ib))
    else:
        option_chains = options(ib)

    # for option_chain in option_chains:
    #     print(option_chain)

    # l = ib.reqSecDefOptParamsAsync("TSLA", "", "STK", underlyingConId)
    await asyncio.gather(option_chains_read, option_chains_options)

if __name__ == "__main__":
    ib = ibs.IB()
    # ib = True
    trader_workstation_port = 7496
    ib.connect("127.0.0.1", trader_workstation_port, clientId=11,
        timeout=20,
        account=bot.conf.ACCOUNT
    )
    asyncio.run(main(ib))

    import sys

    sys.exit(0)
comment = """

run_trading
get contracts of type
suggest options
get expiry and strikes
get option market data
"""
