import asyncio
import datetime
import logging
import time

from numpy import isnan
import ib_insync as ibs

from bot import util
from bot import conf

import loggingx
log_level = logging.DEBUG
logger = loggingx.logger(__file__,log_level)

def suggest_forex(ib):
    contracts = [
        ibs.Forex('EURUSD'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    return contracts


def suggest_fang_stocks(ib):
    contracts = [
        ibs.Stock('AAPL', 'SMART', 'USD'),
        ibs.Stock('AMZN', 'SMART', 'USD'),
        ibs.Stock('FB', 'SMART', 'USD'),
        ibs.Stock('GOOGL', 'SMART', 'USD'),
        ibs.Stock('NFLX', 'SMART', 'USD'),
        ibs.Stock('TSLA', 'SMART', 'USD'),
        ibs.Stock('ZM', 'SMART', 'USD'),
    ]
    return contracts


def suggest_stocks(ib):
    contracts = [
        # ibs.Stock('AAPL', 'SMART', 'USD'),
        ibs.Stock('AMZN', 'SMART', 'USD'),
        ibs.Stock('FB', 'SMART', 'USD'),
        ibs.Stock('GOOGL', 'SMART', 'USD'),

        ibs.Stock('SNOW', 'SMART', 'USD'),
        ibs.Stock('SPY', 'SMART', 'USD'),
        ibs.Stock('QLD', 'SMART', 'USD'),

        # ibs.Stock('AVGO', 'SMART', 'USD'),
        # ibs.Stock('BA', 'SMART', 'USD'),
        # ibs.Stock('BKNG', 'SMART', 'USD'),
        # ibs.Stock('BYND', 'SMART', 'USD'),
        ibs.Stock('CMG', 'SMART', 'USD'),
        ibs.Stock('DIS', 'SMART', 'USD'),
        # ibs.Stock('LRCX', 'SMART', 'USD'),
        ibs.Stock('MA', 'SMART', 'USD'),
        ibs.Stock('MELI', 'SMART', 'USD'),
        ibs.Stock('NFLX', 'SMART', 'USD'),
        ibs.Stock('NOW', 'SMART', 'USD'),
        # ibs.Stock('NTES', 'SMART', 'USD'),
        ibs.Stock('NVDA', 'SMART', 'USD'),
        # ibs.Stock('REGN', 'SMART', 'USD'),
        ibs.Stock('ROKU', 'SMART', 'USD'),
        ibs.Stock('SHOP', 'SMART', 'USD'),
        ibs.Stock('STMP', 'SMART', 'USD'),
        ibs.Stock('SPOT', 'SMART', 'USD'),
        ibs.Stock('TSLA', 'SMART', 'USD'),
        ibs.Stock('TTD', 'SMART', 'USD'),
        ibs.Stock('ZS', 'SMART', 'USD'),
        ibs.Stock('AMD', 'SMART', 'USD'),
        ibs.Stock('NKLA', 'SMART', 'USD'),
        ibs.Stock('ZM', 'SMART', 'USD'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    return contracts


async def suggest_stocks_async(ib):
    contracts = [
        # ibs.Stock('AAPL', 'SMART', 'USD'),
        ibs.Stock('AMZN', 'SMART', 'USD'),
        ibs.Stock('FB', 'SMART', 'USD'),
        ibs.Stock('GOOGL', 'SMART', 'USD'),

        ibs.Stock('SNOW', 'SMART', 'USD'),
        ibs.Stock('SPY', 'SMART', 'USD'),
        ibs.Stock('QLD', 'SMART', 'USD'), #

        # ibs.Stock('AVGO', 'SMART', 'USD'),
        # ibs.Stock('BA', 'SMART', 'USD'),
        # ibs.Stock('BKNG', 'SMART', 'USD'),
        # ibs.Stock('BYND', 'SMART', 'USD'),
        ibs.Stock('CMG', 'SMART', 'USD'),
        ibs.Stock('DIS', 'SMART', 'USD'),
        # ibs.Stock('LRCX', 'SMART', 'USD'),
        # ibs.Stock('MA', 'SMART', 'USD'),
        # ibs.Stock('MELI', 'SMART', 'USD'),
        # ibs.Stock('NFLX', 'SMART', 'USD'),
        # ibs.Stock('NOW', 'SMART', 'USD'),
        # # ibs.Stock('NTES', 'SMART', 'USD'),
        # ibs.Stock('NVDA', 'SMART', 'USD'),
        # # ibs.Stock('REGN', 'SMART', 'USD'),
        # ibs.Stock('ROKU', 'SMART', 'USD'),
        # ibs.Stock('SHOP', 'SMART', 'USD'),
        # ibs.Stock('STMP', 'SMART', 'USD'),
        # ibs.Stock('SPOT', 'SMART', 'USD'),
        # ibs.Stock('TSLA', 'SMART', 'USD'),
        # ibs.Stock('TTD', 'SMART', 'USD'),
        # ibs.Stock('ZS', 'SMART', 'USD'),
        # ibs.Stock('AMD', 'SMART', 'USD'),
        # ibs.Stock('NKLA', 'SMART', 'USD'),
        # ibs.Stock('ZM', 'SMART', 'USD'),
    ]
    # davs2rt does qualifyContractAsync work in parallel witt the list? think so
    # tasks = []
    # for c in contracts:
    #     tasks.append(ib.qualifyContractsAsync(c))
    # foo = await asyncio.gather(*tasks)
    # qcontracts = await ib.qualifyContractsAsync(*contracts)
    logger.debug(contracts)
    begin = time.perf_counter()
    contracts =  await ib.qualifyContractsAsync(*contracts)
    logger.info("qualifying contracts took {:.3f} seconds".format(
        time.perf_counter() - begin))  # 0.234 secs
    logger.info(contracts)

    return contracts


def suggest_all_options(ib):
    stocks = suggest_stocks(ib)

    hack_nflx_min_strike = 450

    options = []
    for stock in stocks:
        chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

        if not chains:
            continue

        target_chain = None
        for chain in chains:
            if chain.exchange != 'SMART':
                continue

            for expiry in chain.expirations:
                for strike in chain.strikes:
                    for right in ['C', 'P']:
                        option = ibs.Option(stock.symbol, expiry, strike, right, 'SMART')
                        if not option:
                            import pdb ; pdb.set_trace()
                        options.append(option)

    qualified = []
    for option in options:
        if not (option.symbol == 'NFLX' and option.strike >= hack_nflx_min_strike):
            continue

        r = ib.qualifyContracts(option)
        if r:
            qualified += r

        logger.debug('q', len(qualified))
        if len(qualified) > 200:
            break

    logger.debug([c.conId for c in qualified])

    return qualified


def suggest_options(ib, limit=None, stocks_limit=None, limit_strike=4, no_filter=False):
    stocks = suggest_stocks_async(ib)[:stocks_limit]
    option_cls = ibs.Option

    options = []

    candidate_options = []
    md_dict = {}
    logger.info('req tickers')
    tickers = ib.reqTickers(*stocks)
    for i, (stock, ticker) in enumerate(zip(stocks, tickers)):
        logger.debug('get chains', stock.localSymbol)
        expiry, strikes = get_expiry_and_strikes(ib, ticker, stock)
        logger.debug('got:', expiry, strikes)
        if not expiry:
            continue

        # logger.debug('strikes', strikes)
        new_candidate_options = []
        for strike in strikes[limit_strike:]:
            for right in ['C', 'P']:
                option = option_cls(stock.symbol, expiry, strike, right, 'SMART')
                new_candidate_options.append(option)

        # logger.debug(new_candidate_options)
        ib.qualifyContracts(*new_candidate_options)
        if not no_filter:
            for option in new_candidate_options:
                md_opt = get_option_market_data(ib, ticker, option)
                logger.debug('->', option.strike, option.right, md_opt)
                md_dict[option.conId] = md_opt
        candidate_options += new_candidate_options

    if no_filter:
        return candidate_options[:limit]

    candidate_tickers = ib.reqTickers(*candidate_options)
    filtered_candidates = []

    for co, ticker in zip(candidate_options, candidate_tickers):
        if not ticker or not ticker.modelGreeks:
            logger.warn('no ticker or greeks for', co)
            continue

        delta = ticker.modelGreeks.delta
        md_dict[co.conId]['delta'] = delta
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']

        if not (abs(delta) > conf.MIN_DELTA and abs(delta) < conf.MAX_DELTA):
            continue

        filtered_candidates.append(co)
        logger.info('candidate\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

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
        logger.info('selected\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    # import pdb ; pdb.set_trace()
    logger.debug('qualify')
    [ib.qualifyContracts(o) for o in candidate_options]
    print('done')
    return candidate_options


async def suggest_options_async(ib, limit=None, stocks_limit=None, limit_strike=4,
        no_filter=False):
    logger.info("get stocks")
    stocks = (await suggest_stocks_async(ib))[:stocks_limit]

    logger.info('get tickers')
    candidate_options = []
    md_dict = {}
    start = time.perf_counter()
    tickers = await ib.reqTickersAsync(*stocks)
    logger.debug(f"ib.reqTickersAsync took {time.perf_counter() - start}")  #5+ secs

    logger.info('get available options')
    loop_start = time.perf_counter()
    strikes_tasks = set()
    i = 0
    for ticker in tickers:
        logger.debug(f'get options for {ticker.contract.localSymbol}')
        strikes_tasks.add(
            asyncio.create_task(get_expiry_and_strikes_async(ib, ticker))
        )
        await asyncio.sleep(0) # let's task get started
        i += 1
    results = await asyncio.gather(*strikes_tasks)
    logger.debug(f"got option chains for {i} securities in "
                 f"{time.perf_counter()-loop_start:0.3f} secs")
    for stock, expiry, strikes in results:
        logger.debug(f'{stock.localSymbol} {expiry}, {strikes}')
        if not expiry:
            continue

        new_candidate_tasks = set()
        start = time.perf_counter()
        for strike in strikes[limit_strike:]:
            for right in ['C', 'P']:
                new_candidate_option = ibs.Option(
                    stock.symbol, expiry, strike, right, 'SMART')
                new_candidate_tasks.add(asyncio.create_task(
                    ib.qualifyContractsAsync(new_candidate_option)))
                await asyncio.sleep(0) # give the new task a chance to run
        new_candidate_options = await asyncio.gather(*new_candidate_tasks)
        # note: this returns a list of lists of options,
        # some lists are empty on strike prices that don't seem to exist
        new_candidate_options = list(filter(lambda x: len(x)!=0, new_candidate_options))
        new_candidate_options = list(map(lambda x: x[0], new_candidate_options))
        logger.debug(f"{stock.symbol} ib.qualifyContractsAsync took"
              f" {time.perf_counter() - start}")  #0.358
        option_data_tasks = set()
        num_options = 0
        start = time.perf_counter()
        if not no_filter:
            for option in new_candidate_options:
                option_data_tasks.add(asyncio.create_task(
                    get_option_market_data_async(ib, ticker, option))
                )
                await asyncio.sleep(0) # let task start
                num_options += 1
            # for option in new_candidate_options:
            #     start = time.perf_counter()
            #     md_opt = await get_option_market_data_async(ib, ticker, option)
            #     logger.debug(f"get_option_market_data_async took {time.perf_counter()-start}")
            #     logger.debug('->', option.strike, option.right, md_opt)
            #     #get_option_market_data_async took 0.0006509020004159538
            #     md_dict[option.conId] = md_opt
        md_opts = await asyncio.gather(*option_data_tasks)

        # BUG BUG need to merge option data values into candidate options
        assert False, "this is where you got frustrated and gave up for the day"

        option_data_dict = []
        for option_data in md_opts:
            option_data_dict[option_data['option']] = dict(
                volume=option_data['volume'],
                open_interest=option_data['open_interest']
            )
        logger.debug(f"gathered {num_options+1} option data records in"
              f" {time.perf_counter()-start:0.3f} secs.")
        candidate_options.extend(new_candidate_options)
        candidate_options = option_data_dict

    logger.info(
        # took 123 seconds with wrapper.error.1014  Warning 10167
        # 2021-09-02 took 17 seconds after creating tasks for expiry and strikes
        f"looping all qualifyContractsAsync took "
        f"{time.perf_counter()-loop_start:0.3f}")
    if no_filter:
        return candidate_options[:limit]

    start = time.perf_counter()
    candidate_tickers = await ib.reqTickersAsync(*option_data_dict.keys())
    # 2021-09-02 ib.reqTickersAsync took 17.015 can this go into the loop as a task
    logger.debug(f"ib.reqTickersAsync took {time.perf_counter() - start}") #6.5 sec
    filtered_candidates = []

    for co, ticker in zip(candidate_options, candidate_tickers):
        if not ticker or not ticker.modelGreeks:
            print('no ticker or greeks for', co)
            continue

        delta = ticker.modelGreeks.delta
        # md_dict[co.conId]['delta'] = delta
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']

        # if isnan(delta) or isnan(v) or isnan(oi):
        #     logger.warning(f"missing data in {stock.symbol}, skipping")
        #     continue
        try:
            if not (abs(delta) > conf.MIN_DELTA and abs(delta) < conf.MAX_DELTA):
                continue
        except TypeError:  #delta can be a nan after hours
            continue

        filtered_candidates.append(co)
        logger.debug('candidate\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

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
        logger.debug('selected\t%s\t%s\t%s\t%s' % (co.localSymbol, v, oi, delta))

    # import pdb ; pdb.set_trace()
    logger.info('qualify')
    candidate_options = [await ib.qualifyContractsAsync(o) for o in candidate_options]
    for co in candidate_options:
        logger.debug(co)
    return candidate_options


def get_option_market_data(ib, ticker, option):
    key = 'market_data_%s' % (option.localSymbol)
    cached = util.get_cached_json(key)
    if cached:
        return cached

    md = ib.reqMktData(option, genericTickList='100,101,104,106')
    tries = 5
    found = False
    while ib.sleep(.1):
        if md.volume > 0 and (md.callOpenInterest + md.putOpenInterest > 0):
            found = True
            break
        tries -= 1
        logger.debug(f'tries {tries}')
        if tries == 0:
            break

    v, coi, poi = (md.volume, md.callOpenInterest, md.putOpenInterest)

    # v, coi, poi = (1, 2, 3)
    md_data = {}
    md_data['volume'] = v
    md_data['open_interest'] = coi + poi

    util.put_cached_json(key, md_data)
    return md_data


async def get_option_market_data_async(ib, ticker, option):
    # key = 'market_data_%s' % (option.localSymbol)
    # cached = util.get_cached_json(key)
    # if cached:
    #     return cached

    md = ib.reqMktData(option, genericTickList='100,101,104,106')
    tries = 6
    found = False
    while True:
        await asyncio.sleep(0.25)
        if md.volume > 0 and (md.callOpenInterest + md.putOpenInterest > 0):
            found = True
            break
        tries -= 1
        # logger.debug(f'tries {tries}')
        if tries == 0:
            break

    v, coi, poi = (md.volume, md.callOpenInterest, md.putOpenInterest)

    # v, coi, poi = (1, 2, 3)
    md_data = {}
    md_data['option'] = option
    md_data['volume'] = v
    md_data['open_interest'] = coi + poi

    # util.put_cached_json(key, md_data)
    return md_data


def get_expiry_and_strikes(ib, ticker, stock):
    key = 'expiry_and_strikes_%s' % (stock.localSymbol)
    cached = util.get_cached_json(key)
    # if cached:
    #     return cached['expiry'], cached['strikes']

    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
    logger.debug('got %s chains' % (len(chains)))

    logger.debug('get market price')
    tries = 5
    while ib.sleep(.1):
        price = ticker.marketPrice()
        if price > 0:
            break
        tries -= 1
        if tries == 0:
            logger.debug(f"couldn't get price, skip it {stock}")
            return None, None
    logger.debug('--> get ticker')
    logger.debug(f'stock {stock}')
    logger.debug(f'price {price}')

    # make sure price is a number
    try:
        _ = price - 1
    except:
        logger.debug('price is nan, skip')
        return None, None

    # logger.debug('chains', chains)
    if not chains:
        return None, None

    target_chain = None
    for chain in chains:
        if chain.exchange != 'SMART':
            continue

        nearest_expiry = min(chain.expirations[1:])
        if not target_chain or min(target_chain.expirations) > min(chain.expirations):
            target_chain = chain

    first_itm_index = None
    for i, s in enumerate(target_chain.strikes):
        expiry = min(target_chain.expirations[1:])
        if s > price:
            first_itm_index = i
            # strike = s
            # break
            break

    n = 10
    strikes = target_chain.strikes[first_itm_index - n:first_itm_index + n + 1]
    util.put_cached_json(key, {'expiry': expiry, 'strikes': strikes})
    return expiry, strikes


async def get_expiry_and_strikes_async(ib, ticker):
    # key = 'expiry_and_strikes_%s' % (stock.localSymbol)
    # cached = util.get_cached_json(key)
    # cached = False  # TODO remove before production
    # if cached:
    #     return cached['expiry'], cached['strikes']
    start = time.perf_counter()
    chains = await ib.reqSecDefOptParamsAsync(
        ticker.contract.symbol, '', ticker.contract.secType, ticker.contract.conId)
    logger.debug(f"getting all option chains (reqSecDefOptParamsAsync) "
                 f"for {ticker.contract.symbol} "
                 f'on {len(chains)} exchanges '
                 f"took {time.perf_counter()-start:0.3f} secs")

    # start filtering out unneeded option chains, starting with exchange
    chains = list(filter(lambda x: x.exchange=='SMART', chains))
    logger.debug(f'{ticker.contract.symbol} SMART chains {chains}')
    if len(chains)==0:
        return None, None, None

    start = time.perf_counter()
    market_price = ticker.marketPrice()
    if market_price > 0: # nan is common, but not > 0
        price = market_price
        price_type = 'market price'
    elif ticker.last > 0:
        price = ticker.last
        price_type = 'last price'
    else:
        logger.warning(f"no available price for {ticker.contract.localSymbol}")
        return None, None, None

    logger.info(f'{ticker.contract.symbol} {price_type} {market_price:0.3f}, '
                f'took {time.perf_counter() - start:0.3f} secs'
    )

    target_chain = None
    for chain in chains:
        # if chain.exchange != 'SMART':
        #     continue

        nearest_expiry = min(chain.expirations[1:])
        if target_chain is None \
        or min(target_chain.expirations) > min(chain.expirations):
            target_chain = chain

    expiry = min(target_chain.expirations[1:])
    first_itm_index = None
    for i, strike in enumerate(target_chain.strikes):
        if strike > price:
            first_itm_index = i
            break
    if first_itm_index is None: # this should never happen
        return None,None, None

    n = 10
    strikes = target_chain.strikes[first_itm_index - n:first_itm_index + n + 1]
    # util.put_cached_json(key, {'expiry': expiry, 'strikes': strikes})
    return ticker.contract, expiry, strikes


def suggest_ranked_options(ib, limit_pull, limit_return):
    options = suggest_options(ib, limit_pull)
    ranking = list(rank_contracts(ib, options).values())
    ranking.sort(key=lambda x: -x['volume'])
    return [r['contract'] for r in ranking[:limit_return]]


def suggest_micro_futures(ib):
    contracts = [
        ibs.Future('MNQ', '20210319', 'GLOBEX'),
        ibs.Future('MES', '20210319', 'GLOBEX'),
        ibs.Future('M2K', '20210319', 'GLOBEX'),
        ibs.Future('MGC', '20210428', 'NYMEX'),

        # ibs.Future('SI', '20210224', 'NYMEX', currency='USD', multiplier=1000),
        # ibs.Future('VXM', '20201021', 'CFE'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    return contracts


def suggest_futures(ib):
    contracts = [
        ibs.Future('NQ', '20210917', 'GLOBEX'),
        ibs.Future('ES', '20210917', 'GLOBEX'),
        ibs.Future('GC', '20210827', 'NYMEX'),
        ibs.Future('SI', '2020928', 'NYMEX', currency='USD', multiplier=5000),
        ibs.Future('RTY', '20210917', 'GLOBEX'),

        # ibs.Future('CL', '20210121', 'NYMEX'),
        # ibs.Future('BRR', '20201127', 'CMECRYPTO'),
        # ibs.Future('VIX', '20201021', 'CFE'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    return contracts


async def suggest_futures_async(ib):
    contracts = [
        ibs.Future('NQ', '20210917', 'GLOBEX'),
        ibs.Future('ES', '20210917', 'GLOBEX'),
        ibs.Future('GC', '20210827', 'NYMEX'),
        ibs.Future('SI', '2020928', 'NYMEX', currency='USD', multiplier=5000),
        ibs.Future('RTY', '20210917', 'GLOBEX'),

        # ibs.Future('CL', '20210121', 'NYMEX'),
        # ibs.Future('BRR', '20201127', 'CMECRYPTO'),
        # ibs.Future('VIX', '20201021', 'CFE'),
    ]
    [await ib.qualifyContractsAsync(c) for c in contracts]
    return contracts


def suggest_futures_options(ib):
    futures = suggest_futures(ib)

    futures_options = []
    for contract in futures:
        chains = ib.reqSecDefOptParams(contract.symbol,
            contract.exchange,
            contract.secType,
            contract.conId
        )
        [ticker] = ib.reqTickers(contract)
        price = ticker.marketPrice()
        target_chain = None
        for chain in chains:
            logger.debug(f"{chain.exchange}, {chain.expirations}")
            logger.debug(chain)
            nearest_expiry = min(chain.expirations)
            if not target_chain or min(target_chain.expirations) > min(chain.expirations):
                target_chain = chain
        for s in target_chain.strikes:
            expiry = min(target_chain.expirations)
            if s > price:
                strike = s
                break
        for right in ['C', 'P']:
            futures_options.append(ibs.FuturesOption(contract.symbol,
                expiry,
                strike,
                right,
                contract.exchange
            )
            )
    return futures_options


async def suggest_futures_options_async(ib):
    futures = await suggest_futures_async(ib)

    futures_options = []
    for contract in futures:
        chains = await ib.reqSecDefOptParamsAsync(contract.symbol,
            contract.exchange,
            contract.secType,
            contract.conId
        )
        [ticker] = await ib.reqTickersAsync(contract)
        price = ticker.marketPrice()
        target_chain = None
        for chain in chains:
            # logger.debug(f"{chain.exchange}, {chain.expirations}")
            logger.debug(chain)
            nearest_expiry = min(chain.expirations)
            if not target_chain or min(target_chain.expirations) > min(chain.expirations):
                target_chain = chain
        if target_chain is None:
            continue
            logger.warning("whoops, what's going on")
        for s in target_chain.strikes:
            expiry = min(target_chain.expirations)
            if s > price:
                strike = s
                break
        for right in ['C', 'P']:
            futures_options.append( ibs.FuturesOption(contract.symbol,
                expiry,
                strike,
                right,
                contract.exchange
                )
            )
    return futures_options


def rank_contracts(ib, contracts):
    def get_volume(ib, contract):
        ib_bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime=None,
            durationStr='2 D',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=True)
        use_bars = ib_bars[-8:]
        return sum([b.volume for b in use_bars]), None

        # md = ib.reqMktData(contract)
        # attempts_left = 5
        # while True:
        #     volume = md.volume
        #     mp, bid, ask = md.midpoint(), md.bid, md.ask
        #     if 'nan' not in [str(x) for x in [volume, mp, bid, ask]]:
        #         spread = ask - float(bid)
        #         pct = spread / mp
        #         return volume, pct
        #     attempts_left -= 1
        #     if not attempts_left:
        #         return get_yesterday_volume(ib, contract)
        #     ib.sleep(.1)

    results = {}
    for contract in contracts:
        ib.qualifyContracts(contract)
        volume, pct = get_volume(ib, contract)
        logger.debug(f'volume {contract.symbol}, {volume}')
        results[contract.conId] = {
            'contract': contract,
            'symbol': contract.symbol,
            'local_symbol': contract.localSymbol,
            'spread_percent': pct,
            'volume': volume,
        }

    return results


class OptionsWatchlist:
    def __init__(self, ib, account):
        self.ib = ib
        self.account = account

    def candidates(self):
        stocks = suggest_stocks(self.ib)
        tickers = self.ib.reqTickers(*stocks)
        self.ib.sleep(.01)
        candidates = []
        for i in range(len(stocks)):
            stock, ticker = [l[i] for l in [stocks, tickers]]
            expiry, strikes = get_expiry_and_strikes(self.ib, ticker, stock)
            logger.debug(f"{stock.symbol}, {ticker.last}, {expiry}, {strikes}")

            for strike in strikes:
                for right in ['C', 'P']:
                    option = ibs.Option(stock.symbol, expiry, strike, right, 'SMART')
                    candidates.append(option)
        candidates = candidates[:50]
        self.ib.qualifyContracts(*candidates)
        return candidates

    # def _get_volume(self, contract):


class Crypto:
    def __init__(self, pair):
        self.pair = pair

    def __repr__(self):
        return '<Crypto: %s>' % self.pair

    @property
    def localSymbol(self):
        return 'Crypto-%s' % self.pair

    @property
    def secType(self):
        return 'CRYPTO'

    @property
    def conId(self):
        return self.localSymbol


def suggest_crypto():
    return [
        Crypto('BTC-USD'),
        # Crypto('LTC-USD'),
        # Crypto('ETH-USD'),
        # Crypto('BCH-USD'),
        # Crypto('XRP-USD'),
    ]
