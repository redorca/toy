import asyncio
import datetime
import logging
import time

from numpy import isnan
import ib_insync as ibs

from bot import util
from bot import conf

# logging.basicConfig(
#     level=logging.WARNING,
#     format='%(asctime)s.%(msecs).3d [%(levelname).3s] %(module)s - %(funcName)s: %('
#            'message)s',
#     # datefmt='%Y-%m-%d %H:%M:%S',
#     datefmt='%H:%M:%S',
# )

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
        ibs.Stock('QLD', 'SMART', 'USD'),

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
    logging.debug(contracts)
    begin = time.perf_counter()
    contracts =  await ib.qualifyContractsAsync(*contracts)
    logging.info("qualifying contracts took {:.3f} seconds".format(
        time.perf_counter() - begin))  # 0.234 secs
    logging.info(contracts)

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

        print('q', len(qualified))
        if len(qualified) > 200:
            break

    print([c.conId for c in qualified])

    return qualified


def suggest_options(ib, limit=None, stocks_limit=None, limit_strike=4, no_filter=False):
    stocks = suggest_stocks_async(ib)[:stocks_limit]
    option_cls = ibs.Option

    options = []

    candidate_options = []
    md_dict = {}
    print('req tickers')
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


async def suggest_options_async(ib, limit=None, stocks_limit=None, limit_strike=4,
        no_filter=False):
    stocks = (await suggest_stocks_async(ib))[:stocks_limit]

    options = []
    candidate_options = []
    md_dict = {}
    print('req tickers')
    start = time.perf_counter()
    tickers = await ib.reqTickersAsync(*stocks)
    print(f"ib.reqTickersAsync took {time.perf_counter() - start}")  #5+ secs

    # TODO I *think* we want to set each of these expiry&strikes call as a
    # separate task, and launch them with a collect or as_completed.
    loop_start = time.perf_counter()
    for i, (stock, ticker) in enumerate(zip(stocks, tickers)):
        print('get chains', stock.localSymbol)
        start = time.perf_counter()
        expiry, strikes = await get_expiry_and_strikes_async(ib, ticker, stock)
        print(f"expiry & strikes takes {time.perf_counter()-start} secs") # 0.005
        print('got:', expiry, strikes)
        if not expiry:
            continue

        # print('strikes', strikes)
        new_candidate_options = []
        for strike in strikes[limit_strike:]:
            for right in ['C', 'P']:
                option = ibs.Option(stock.symbol, expiry, strike, right, 'SMART')
                new_candidate_options.append(option)

        # print(new_candidate_options)
        # start = time.perf_counter()
        new_candidate_options = await ib.qualifyContractsAsync(*new_candidate_options)
        print(f"ib.qualifyContractsAsync took {time.perf_counter() - start}") #0.358
        if not no_filter:
            for option in new_candidate_options:
                start = time.perf_counter()
                md_opt = await get_option_market_data_async(ib, ticker, option)
                print(f"get_option_market_data_async took {time.perf_counter()-start}")
                print('->', option.strike, option.right, md_opt)
                #get_option_market_data_async took 0.0006509020004159538
                md_dict[option.conId] = md_opt
        candidate_options += new_candidate_options

    logging.info( # took 123 seconds with wrapper.error.1014  Warning 10167
        f"looping all qualifyContractsAsync took {time.perf_counter()-loop_start}")
    if no_filter:
        return candidate_options[:limit]

    start = time.perf_counter()
    candidate_tickers = await ib.reqTickersAsync(*candidate_options)
    print(f"ib.reqTickersAsync took {time.perf_counter() - start}") #6.5 sec
    filtered_candidates = []

    for co, ticker in zip(candidate_options, candidate_tickers):
        if not ticker or not ticker.modelGreeks:
            print('no ticker or greeks for', co)
            continue

        delta = ticker.modelGreeks.delta
        md_dict[co.conId]['delta'] = delta
        v = md_dict[co.conId]['volume']
        oi = md_dict[co.conId]['open_interest']

        # if isnan(delta) or isnan(v) or isnan(oi):
        #     logging.warning(f"missing data in {stock.symbol}, skipping")
        #     continue
        try:
            if not (abs(delta) > conf.MIN_DELTA and abs(delta) < conf.MAX_DELTA):
                continue
        except TypeError:  #delta can be a nan after hours
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
    candidate_options = [await ib.qualifyContractsAsync(o) for o in candidate_options]
    for co in candidate_options:
        print(co)
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
        print('tries', tries)
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
    tries = 5
    found = False
    while True:
        await asyncio.sleep(0.1)
        if md.volume > 0 and (md.callOpenInterest + md.putOpenInterest > 0):
            found = True
            break
        tries -= 1
        print('tries', tries)
        if tries == 0:
            break

    v, coi, poi = (md.volume, md.callOpenInterest, md.putOpenInterest)

    # v, coi, poi = (1, 2, 3)
    md_data = {}
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
    print('got %s chains' % (len(chains)))

    print('get market price')
    tries = 5
    while ib.sleep(.1):
        price = ticker.marketPrice()
        if price > 0:
            break
        tries -= 1
        if tries == 0:
            print("couldn't get price, skip it", stock)
            return None, None
    print('--> get ticker')
    print('stock', stock)
    print('price', price)

    # make sure price is a number
    try:
        _ = price - 1
    except:
        print('price is nan, skip')
        return None, None

    # print('chains', chains)
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


async def get_expiry_and_strikes_async(ib, ticker, stock):
    key = 'expiry_and_strikes_%s' % (stock.localSymbol)
    cached = util.get_cached_json(key)
    cached = False  # TODO remove before production
    if cached:
        return cached['expiry'], cached['strikes']
    start = time.perf_counter()
    chains = await ib.reqSecDefOptParamsAsync(stock.symbol, '', stock.secType,
        stock.conId)
    logging.debug(f"reqSecDefOptParamsAsync for {stock.symbol} took "
                  f"{time.perf_counter()-start} secs")
    logging.info(f'got {len(chains)} chains')

    print('get market price')
    tries = 5

    ticker = ib.reqMktData(stock, "", True)
    while True:
        await asyncio.sleep(.1)
        price = ticker.last
        if price > 0:
            break
        tries -= 1
        if tries == 0:
            print("couldn't get price, skip it", stock)
            return None, None


    while False:  # this wasn't working for me to generate a price
        price = ticker.marketPrice()
        price = ticker.last
        # price = 2726
        if price > 0:
            break
        tries -= 1
        if tries == 0:
            print("couldn't get price, skip it", stock)
            return None, None
        await asyncio.sleep(.51)

    print('--> get ticker')
    print('stock', stock)
    print('price', price)

    # make sure price is a number
    # try:
    #     _ = price - 1
    # except:
    #     print('price is nan, skip')
    #     return None, None
    if type(price) not in (int, float):
        return None, None

    # print('chains', chains)
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


def suggest_futures_options(ib):
    futures = suggest_futures(ib)

    futures_options = []
    for contract in futures:
        chains = ib.reqSecDefOptParams(contract.symbol, contract.exchange, contract.secType, contract.conId)
        [ticker] = ib.reqTickers(contract)
        price = ticker.marketPrice()
        target_chain = None
        for chain in chains:
            print(chain.exchange, chain.expirations)
            print(chain)
            nearest_expiry = min(chain.expirations)
            if not target_chain or min(target_chain.expirations) > min(chain.expirations):
                target_chain = chain
        for s in target_chain.strikes:
            expiry = min(target_chain.expirations)
            if s > price:
                strike = s
                break
        for right in ['C', 'P']:
            futures_options.append(ibs.FuturesOption(contract.symbol, expiry, strike, right, contract.exchange))
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
        print('volume', contract.symbol, volume)
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
            print(stock.symbol, ticker.last, expiry, strikes)

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
