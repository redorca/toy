"""this is just a shim I use to call main and be able to debug in pycharm"""
import asyncio
import logging
import pathlib

import loggingx
import main

logging.basicConfig(level=logging.WARNING)
logger = loggingx.logger(__file__, logging.DEBUG)


result = asyncio.run(
    main.run_trading_async(
        contract_type=("options", "futures_options")[0],
        backtest=False,
        ignore_market_data=True,
        limit=None,
        use_obv=True,
        use_ema=False,
    ),
    debug=False,
)

call_chain = """
main:run_trading ib.connect
main:get_contracts_of_type
contracts:suggest_options = ib.reqTickers ->Async; ib.qualifyContracts->Async
contracts:suggest_stocks = ib.qualifyContracts->Async
contracts:get_expiry_and_strikes = ib.reqSecDefOptParams->Async
contracts:marketPrice (no Async) 

start = time.perf_counter()
print(f" took {time.perf_counter() - start}")

"""
