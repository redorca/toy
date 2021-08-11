import asyncio
import time
import ib_insync as ibs

import nest_asyncio
nest_asyncio._patch_asyncio()

import bot.conf

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

            option_chain =  await ib.reqSecDefOptParamsAsync(symbol, '', sec_type, conId)
            option_chains.append(option_chain)
            # print("inner loop time {}".format(time.perf_counter()-start))
    return option_chains


if __name__ == "__main__":
    ib = ibs.IB()
    trader_workstation_port = 7496
    ib.connect("127.0.0.1", trader_workstation_port, clientId=11, timeout=20,
        account=bot.conf.ACCOUNT)
    # loop = asyncio.get_event_loop()
    stocks = []

    start = time.perf_counter()
    if False:
        option_chains =  asyncio.run(options_async(ib))
    else:
        option_chains = options(ib)
    for option_chain in option_chains:
        print(option_chain)
    end = time.perf_counter()
    print("that took {} secs".format(end-start))
    # l = ib.reqSecDefOptParamsAsync("TSLA", "", "STK", underlyingConId)

    import sys
    sys.exit(0)
comment = """

run_trading
get contracts of type
suggest options
get expiry and strikes
get option market data
"""
