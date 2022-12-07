'''
starting with simple2, I want to experiment with ib's loop vs. the one I start.
davs2rt
'''
import asyncio
import time
import ib_insync as ibs

import nest_asyncio

nest_asyncio._patch_asyncio()

import bot.conf

# in_queue = asyncio.Queue(1000)
# out_queue = asyncio.Queue(1000)
options_q = asyncio.Queue(100)


def options(ib):
    begin = time.perf_counter()
    option_chains = []
    for position in ib.positions(bot.conf.ACCOUNT):
        if isinstance(position.contract, ibs.Stock):
            start = time.perf_counter()
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            foo = ib.reqSecDefOptParams(symbol, '', sec_type, conId)
            option_chains.append(foo)
            end = time.perf_counter()
            print("fetching {} took {:.1f} seconds".format(symbol, end-start))
    print("whole options pull took {:.1f} seconds".format(end - begin))
    return option_chains


async def options_async(ib):
    begin = time.perf_counter()
    positions = ib.positions(bot.conf.ACCOUNT)
    print("fetching all positions took {:e} secs".format(time.perf_counter() - begin))
    option_chains = []
    futures = []
    for position in positions:
        if isinstance(position.contract, ibs.Stock):
            start = time.perf_counter()
            print('stock:', position.contract)
            symbol = position.contract.symbol
            conId = position.contract.conId
            sec_type = 'STK'
            # if we created tasks could we run these concurrently?
            futures.append(ib.reqSecDefOptParamsAsync(symbol, '', sec_type,
                conId,))
            # print("option chain:", option_chains[:-1])
            end = time.perf_counter()
            print("fetching {} option chain took {:.3f} seconds".format(symbol,
                end - start))
    for chain in  asyncio.as_completed(futures):
        waiting_start = time.perf_counter()
        await options_q.put(chain)
        print("data wait  took {:.3f} seconds".format(
            time.perf_counter() - waiting_start))
        await asyncio.sleep(0.1) # yield to queue
    print("start fetching ALL options took {:.3f} seconds".format(time.perf_counter(

    )-begin))


async def read_task():
    begin = time.perf_counter()
    total_fake_delays = 0
    try:
        while True:
            start = time.perf_counter()
            # await asyncio.sleep(1)
            print("{} items in queue".format(options_q.qsize()))
            option_chain_print = await options_q.get()
            end = time.perf_counter()
            options_q.task_done()
            print(await option_chain_print)
            # print("queue read took {:.3f} secs".format(end - start))
            # print("{} items in queue".format(options_q.qsize()))
            extra_processing = False
            if extra_processing:
                j = 0
                st = time.perf_counter()
                for i in range(int(15000)):
                    j += 1
                    await asyncio.sleep(0)
                faked_delay = time.perf_counter() - st
                total_fake_delays += faked_delay
                print("faked delay added {} secs".format(faked_delay))
    except asyncio.CancelledError:
        raise
    finally:
        print("q reader ended, total faked processing delay was {} secs".format(
            total_fake_delays))



async def main(ib):
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    read_tsk = asyncio.create_task(read_task(), name="q reader")
    options_tsk = asyncio.create_task(options_async(ib), name='option fetcher')
    done, pending = await asyncio.wait([read_tsk, options_tsk],
        return_when=asyncio.FIRST_COMPLETED)
    print("{} are done".format(done))
    while options_q.qsize() > 0:
        # print("queue has {} items to process".format(options_q.qsize()))
        await asyncio.sleep(0.1)
    for task in pending:
        print("cancelling", task.get_name())
        task.cancel()
    print('entire process took {:.1f} seconds'.format(time.perf_counter()-start))


if __name__ == "__main__":
    ib = ibs.IB()
    main_loop = asyncio.get_event_loop()
    # print(asyncio.get_running_loop())
    print(main_loop)
    try:
        print(asyncio.get_running_loop())
    except RuntimeError:
        pass # no running loop, but now one exists.
    # ib = True
    trader_workstation_port = 7496
    ib.connect("127.0.0.1", trader_workstation_port, clientId=11,
               timeout=20,
               account=bot.conf.ACCOUNT
               )
    print("starting main loop")
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
