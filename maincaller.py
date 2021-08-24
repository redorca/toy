"""this is just a trick I use to call main and be able to debug in pycharm"""
import asyncio
import src.main

print("main caller")

result = asyncio.run(
    src.main.run_trading_async(
        contract_type="options",
        backtest=False,
        ignore_market_data=True,
        limit=None,
        use_obv=True,
        use_ema=False,
    ),
    debug=True,
)

call_chain = """
main:run_trading ib.connect
main:get_contracts_of_type
contracts:suggest_options = ib.reqTickers ->Async; ib.qualifyContracts->Async
contracts:suggest_stocks = ib.qualifyContracts->Async
contracts:get_expiry_and_strikes = ib.reqSecDefOptParams->Async
contracts:marketPrice (no Async) 
"""
