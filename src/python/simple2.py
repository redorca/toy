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
    for position in ib.positions(bot.conf.ACCOUNT):
        start = time.perf_counter()
        if isinstance(position.contract, ibs.Stock):
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            option_chain = await ib.reqSecDefOptParamsAsync(symbol, '', sec_type, conId)
            await in_queue.put(option_chain)


async def read_task():
    start = time.perf_counter()
    while True:
        print("waiting for data...")
        option_chain_print = await in_queue.get()
        end = time.perf_counter()
        in_queue.task_done()
        print("ASYNC that took {} secs \n", format(end - start))
        # print(option_chain_print, in_queue.qsize())


async def main(ib):
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    if True:
        option_chains_read = loop.create_task(read_task())
        option_chains_options = loop.create_task(options_async(ib))
        option_chains_options1 = loop.create_task(options_async(ib))
        option_chains_options2 = loop.create_task(options_async(ib))
        option_chains_options3 = loop.create_task(options_async(ib))
        option_chains_options4 = loop.create_task(options_async(ib))

    await asyncio.gather(option_chains_read, option_chains_options, option_chains_options1, option_chains_options2,
                         option_chains_options4, option_chains_options3)


if __name__ == "__main__":
    print("starting main loop")
    ib = ibs.IB()
    # ib = True
    trader_workstation_port = 7496
    ib.connect("127.0.0.1", trader_workstation_port, clientId=11,
               timeout=20,
               account=bot.conf.ACCOUNT
               )
    print("starting main loop")
    start = time.perf_counter()
    option_chains = options(ib)
    print("SYNC took {} secs \n", format(time.perf_counter() - start))
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
