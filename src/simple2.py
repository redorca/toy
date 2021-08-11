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
    for position in ib.positions(bot.conf.ACCOUNT):
        start = time.perf_counter()
        if isinstance(position.contract, ibs.Stock):
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            option_chain = await ib.reqSecDefOptParamsAsync(symbol, '', sec_type, conId)
            in_queue.put_nowait(option_chain)
            # option_chains.append(option_chain)
            # print("inner loop time {}".format(time.perf_counter()-start))


async def read_task():
    while True:
        option_chain_print = in_queue.get()
        print(option_chain_print)


async def awaiting_for_complete(event1, event2):
    await asyncio.gather(event1, event2)


def main():
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    # loop.run_until_complete(main())
    ib = ibs.IB()
    trader_workstation_port = 7496
    ib.connect("127.0.0.1", trader_workstation_port, clientId=11, timeout=20,
               account=bot.conf.ACCOUNT)
    start = time.perf_counter()
    if True:
        option_chains_read = asyncio.create_task(read_task, loop=asyncio.new_event_loop())
        # create a queue of data.
        option_chains_options = asyncio.create_task(options_async(ib), loop=asyncio.new_event_loop())
        # option_chains = asyncio.run(options_async(ib))
    else:
        option_chains = options(ib)

    # for option_chain in option_chains:
    #     print(option_chain)
    end = time.perf_counter()
    print("that took {} secs".format(end - start))
    # l = ib.reqSecDefOptParamsAsync("TSLA", "", "STK", underlyingConId)
    loop.run_until_complete(awaiting_for_complete(option_chains_read, option_chains_options))


if __name__ == "__main__":
    main()
    import sys

    sys.exit(0)
comment = """

run_trading
get contracts of type
suggest options
get expiry and strikes
get option market data
"""
