import time
import requests
import requests_cache
import datetime

from bot.provider import Bar


requests_cache.install_cache('crypto')


class CryptoConnection:
    def qualifyContracts(*args, **kwargs):
        pass

    def reqHistoricalData(self, contract=None, endDateTime=None, durationStr='1 D',
            barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True):
        if endDateTime is None:
            endDateTime = datetime.datetime.now()

        bar_size = barSizeSetting
        end_dt = endDateTime

        parts = durationStr.split(' ')
        num_days = int(parts[0])
        if parts[1] != 'D':
            raise Exception('unexpected duration str %s' % durationStr)
        start_dt = end_dt - datetime.timedelta(days=num_days)

        return coinbase_bars(contract.pair, bar_size, start_dt, end_dt)

    def sleep(self, secs):
        time.sleep(secs)
        return True

    def openTrades(self, *args, **kwargs):
        # TODO
        return []

    def reqContractDetails(self, contract, **kwargs):
        # TODO
        return CryptoContractDetails(contract)


class CryptoContractDetails:
    def __init__(self, contract):
        self.contract = contract


def coinbase_bars(pair, bar_size, start_dt, end_dt):
    granularity = {
        '1 min': 60,
        '5 mins': 300,
    }[bar_size]

    start_dt -= datetime.timedelta(days=1)
    use_end_dt = end_dt + datetime.timedelta(days=1)
    bars = []
    retries = 5
    while start_dt < use_end_dt:
        batch_end_dt = start_dt + datetime.timedelta(hours=6)

        url = 'https://api.pro.coinbase.com/products/%s/candles?granularity=%s&start=%s&end=%s' % (pair, granularity, start_dt.isoformat(), batch_end_dt.isoformat())
        # print(url)
        resp = requests.get(url)
        bars = []
        data = resp.json()
        if 'message' in data:
            print(data['message'])
            if retries <= 0:
                raise Exception(data['message'])
            time.sleep(1)
            retries -= 1
            continue

        for r in data:
            t, low, high, o, close, volume = r
            bars.append(Bar(o, high, low, close, volume, datetime.datetime.fromtimestamp(t), None))
        start_dt = batch_end_dt
        retries += 1
    return bars
    
