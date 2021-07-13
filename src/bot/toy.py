"""toy application to investigate speed improvements from async with ib-"""
import asyncio
import datetime
import time

import ib_insync as ibs

from bot import util
from bot import conf
from bot import connect
from bot.contracts import suggest_stocks, \
    suggest_stocks_async, get_option_market_data, get_expiry_and_strikes

def elapsed_time(start:datetime.datetime):
    now_ = datetime.datetime.now()
    return now_ - start

async def suggest_options_async(ib, limit=None, stocks_limit=None, limit_strike=4,
    no_filter=False):
    stocks = suggest_stocks_async(ib)[:stocks_limit]
    option_cls = ibs.Option

    options = []

    candidate_options = []
    md_dict = {}
    print('req tickers')
    # async davs2rt
    tickers = await ib.reqTickersAsync(*stocks)
    for i, (stock, ticker) in enumerate(zip(stocks, tickers)):
        print('get chains', stock.localSymbol)
        expiry, strikes = get_expiry_and_strikes(ib, ticker, stock)
        print('got:', expiry, strikes)
        if not expiry:
            continue

        # print('strikes', strikes)
        new_candidate_options = []
        for strike in strikes[limit_strike:]:
            for right in ['C', 'P']:
                option = option_cls(stock.symbol, expiry, strike, right, 'SMART')
                new_candidate_options.append(option)

        # print(new_candidate_options)
        ib.qualifyContracts(*new_candidate_options)
        if not no_filter:
            for option in new_candidate_options:
                md_opt = get_option_market_data(ib, ticker, option)
                print('->', option.strike, option.right, md_opt)
                md_dict[option.conId] = md_opt
        candidate_options += new_candidate_options

    if no_filter:
        return candidate_options[:limit]
    # async davs2rt
    candidate_tickers = ib.reqTickers(*candidate_options)
    filtered_candidates = []

    for co, ticker in zip(candidate_options, candidate_tickers):
        if not ticker or not ticker.modelGreeks:
            print('no ticker or greeks for', co)
            continue

        delta = ticker.modelGreeks.delta
        md_dict[co.conId]['delta'] = delta
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']

        if not (abs(delta) > conf.MIN_DELTA and abs(delta) < conf.MAX_DELTA):
            continue

        filtered_candidates.append(co)
        print('candidate\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    candidate_options = filtered_candidates
    # candidate_options.sort(key=lambda x:abs(.5 - abs(md_dict[x.conId]['delta'])))
    # candidate_options = candidate_options[:limit * 4]
    candidate_options.sort(key=lambda x: -md_dict[x.conId]['open_interest'])
    candidate_options = candidate_options[:limit * 2]
    candidate_options.sort(key=lambda x: -md_dict[x.conId]['volume'])
    candidate_options = candidate_options[:limit]

    for co in candidate_options:
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']
        delta = md_dict[co.conId]['delta']
        print('selected\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    # import pdb ; pdb.set_trace()
    print('qualify')
    [ib.qualifyContracts(o) for o in candidate_options]
    print('done')
    return candidate_options


def suggest_options(ib, limit=None, stocks_limit=None, limit_strike=4, no_filter=False):
    stocks = suggest_stocks(ib)[:stocks_limit]
    option_cls = ibs.Option

    options = []

    candidate_options = []
    md_dict = {}
    print('req tickers')
    # async davs2rt
    tickers = ib.reqTickers(*stocks)
    for i, (stock, ticker) in enumerate(zip(stocks, tickers)):
        print('get chains', stock.localSymbol)
        expiry, strikes = get_expiry_and_strikes(ib, ticker, stock)
        print('got:', expiry, strikes)
        if not expiry:
            continue

        # print('strikes', strikes)
        new_candidate_options = []
        for strike in strikes[limit_strike:]:
            for right in ['C', 'P']:
                option = option_cls(stock.symbol, expiry, strike, right, 'SMART')
                new_candidate_options.append(option)

        # print(new_candidate_options)
        ib.qualifyContracts(*new_candidate_options)
        if not no_filter:
            for option in new_candidate_options:
                md_opt = get_option_market_data(ib, ticker, option)
                print('->', option.strike, option.right, md_opt)
                md_dict[option.conId] = md_opt
        candidate_options += new_candidate_options

    if no_filter:
        return candidate_options[:limit]
    # async davs2rt
    candidate_tickers = ib.reqTickers(*candidate_options)
    filtered_candidates = []

    for co, ticker in zip(candidate_options, candidate_tickers):
        if not ticker or not ticker.modelGreeks:
            print('no ticker or greeks for', co)
            continue

        delta = ticker.modelGreeks.delta
        md_dict[co.conId]['delta'] = delta
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']

        if not (abs(delta) > conf.MIN_DELTA and abs(delta) < conf.MAX_DELTA):
            continue

        filtered_candidates.append(co)
        print('candidate\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    candidate_options = filtered_candidates
    # candidate_options.sort(key=lambda x:abs(.5 - abs(md_dict[x.conId]['delta'])))
    # candidate_options = candidate_options[:limit * 4]
    candidate_options.sort(key=lambda x: -md_dict[x.conId]['open_interest'])
    candidate_options = candidate_options[:limit * 2]
    candidate_options.sort(key=lambda x: -md_dict[x.conId]['volume'])
    candidate_options = candidate_options[:limit]

    for co in candidate_options:
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']
        delta = md_dict[co.conId]['delta']
        print('selected\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    # import pdb ; pdb.set_trace()
    print('qualify')
    [ib.qualifyContracts(o) for o in candidate_options]
    print('done')
    return candidate_options


def main():
    start = datetime.datetime.now()
    ib = connect.connect()
    contracts = suggest_stocks(ib)
    fin = datetime.datetime.now( )
    for c in contracts:
        print(c)
    print(fin - start)
    return ib



async def main_async(ib):
    start = datetime.datetime.now()
    # ib = await connect.connect_async()
    if not ib.isConnected():
        print('not connected')
        raise RuntimeError("7f")
    contracts = await suggest_stocks_async(ib)
    fin = datetime.datetime.now()
    ib.disconnect()
    for c in contracts:
        print(c)
    print(fin - start)


if __name__ == '__main__':
    print("synchronous")
    ib = main()

    print("asynchronous")
    asyncio.run(main_async(ib), debug=True)
    print('done')
