import requests
import os
import json
import datetime
import pytz
import ib_insync as ibs
import pickle

from diskcache import Cache
from bot import util

cache = Cache('diskcache')


def contract_pretty(contract):
    if contract.secType == 'OPT':
        return '%s: %s@%s %s' % (contract.symbol, contract.lastTradeDateOrContractMonth, contract.strike, contract.right)
    else:
        return str(contract)


def is_options_market_data_available(ib):
    from bot.contracts import suggest_options

    contracts = suggest_options(ib, limit=1, stocks_limit=1, no_filter=True)
    print(contracts[0])
    md = ib.reqMktData(contracts[0])
    ib.sleep(1)
    print('options data?', md.volume)
    return md.volume > 0


def put_cached_pickle(key, bars):
    global cache
    # print('put check key pickle [%s]' % key)
    expire = int(datetime.datetime.now().timestamp()) + 30*24*3600
    bytestr = pickle.dumps(bars)
    cache.set(key, bytestr, expire=expire)


def get_cached_pickle(key):
    global cache
    # print('get check key pickle [%s]' % key)
    # import pdb ; pdb.set_trace()
    if key in cache:
        return pickle.loads(cache[key])
    return None


def put_cached_json(key, data):
    global cache
    # print('store cache', key)
    expire = util.seconds_to_next_market_open() + 3600
    cache.set(key, json.dumps(data), expire=expire)


def get_cached_json(key):
    global cache
    if key in cache:
        return json.loads(cache[key])
    return None


def is_before_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open = datetime.datetime(year=now.year, month=now.month, day=now.day,
                                    hour=9, minute=30, second=0).astimezone(eastern)
    return now < market_open


def is_before_market_close():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_close = datetime.datetime(year=now.year, month=now.month, day=now.day,
                                     hour=16, minute=0, second=0).astimezone(eastern)
    return now < market_close


# def seconds_to_previous_market_open():
#     eastern = pytz.timezone('US/Eastern')
#     now = datetime.datetime.now(eastern)
#     prev_open = datetime.datetime(year=now.year, month=now.month, day=now.day,
#                                   hour=9, minute=30, second=0).astimezone(eastern)
#     if now.hour < 9 or now.hour == 9 and now.minute < 30:
#         next_open -= datetime.timedelta(days=1)
#     return (next_open - now).seconds


def seconds_to_next_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    next_open = datetime.datetime(year=now.year, month=now.month, day=now.day,
                                  hour=9, minute=30, second=0).astimezone(eastern)
    if now.hour > 9 or now.hour == 9 and now.minute > 30:
        next_open += datetime.timedelta(days=1)
    return (next_open - now).seconds


def seconds_to_next_market_close():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    next_close = datetime.datetime(year=now.year, month=now.month, day=now.day,
                                   hour=4, minute=0, second=0).astimezone(eastern)
    if now.hour > 4:
        next_close += datetime.timedelta(days=1)
    return (next_close - now).seconds


def flatten_positions(ib, account):
    positions = ib.positions()  # A list of positions, according to IB
    for position in positions:
        if position.account != account:
            print('skip position in account %s' % position.account)
            continue

        print('position', position)
        contract = position.contract
        ib.qualifyContracts(contract)
        totalQuantity = abs(position.position)
        if position.position > 0: # Number of active Long positions
            action = 'SELL' # to offset the long positions
        elif position.position < 0: # Number of active Short positions
            action = 'BUY' # to offset the short positions
        else:
            assert False

        order = ibs.MarketOrder(action, totalQuantity, account=account, tif='GTC',
                                outsideRth=True)
        print('order', order)
        trade = ib.placeOrder(contract, order)
        print('trade', trade)
        print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
        assert trade in ib.trades(), 'trade not listed in ib.trades'


def parse_date_str(date_str, hour=0):
    year, month, day = [int(s) for s in date_str.split('-')]
    return datetime.datetime(year=year, month=month, day=day,
                             hour=hour, minute=0, second=0)


def slack_message(msg):
    url = 'https://hooks.slack.com/services/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'
    requests.post(url, data=json.dumps({'text': msg}))


def last_bid_ask(ib, contract):
    details = ib.reqContractDetails(contract)
    [ticker] = ib.reqTickers(contract)
    bid, ask = None, None
    max_tries = 10
    while ask is None and ib.sleep(.01) and max_tries > 0:
        max_tries -= 1
        ask = ticker.ask
        bid = ticker.bid
    return bid, ask


def min_increment(ib, contract):
    ib.qualifyContracts(contract)
    details = ib.reqContractDetails(contract)
    [ticker] = ib.reqTickers(contract)
    return details[0].minTick

def dump_dict(source,/, Color="34"):
    Prefix = ''.join(["\033[1;", Color, "m"])
    Reset = "\033[0;39;49m"
    distance = [ '\t', '\t', '\t', '\t', '\t', '\t',]
    '''
        list each dictionary label and value in an aligned format
    '''
    for val in source:
        xtent = (39 - len(val)) / 8
        space = ''.join(distance[:int(xtent)])
        print(f"\t{Prefix}{val}{Reset}{space}{source[val]}")
    print("\n\n")
