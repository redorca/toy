import datetime
import matplotlib.pyplot as plt
import numpy as np

from bot import analyzer
from bot import util
from bot import conf


def get_ranges(ib, contract, bar_size, field, num_days):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=num_days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    # percentile_buckets = [1, 2, 5, 10, 15, 20, 25, 50, 70, 80, 90, 95, 98, 99]
    percentile_buckets = []
    n = conf.PERCENTILE_WIDTH
    for i in range(100):
        percentile_buckets.append((int(i/n)) * n)
    percentile_buckets = list(set(percentile_buckets))
    percentile_buckets.sort()
    # print('percentile_buckets', percentile_buckets)
    # import pdb ; pdb.set_trace()

    ib.qualifyContracts(contract)

    cache_key = 'ranges_' + '_'.join([str(x) for x in [
        contract.localSymbol, start_date_str, end_date_str, bar_size,
        field, num_days, ':'.join([str(q) for q in percentile_buckets])]])

    # print('ranges cache key', cache_key)
    
    cached = util.get_cached_json(cache_key)
    if cached is not None:
        return cached

    df = analyzer.pull_training_data(ib, contract, start_date_str, end_date_str, bar_size=bar_size)

    arr = df[field]
    print(df)
    arr = arr[arr.notnull()]
    ranges = {}
    print('arr', arr)

    for q in percentile_buckets:
        v = np.percentile(arr, q)
        print('percentile', q, v)
        # import pdb ; pdb.set_trace()
        ranges['p%02d' % q] = {
            'min_value': v
        }

    util.put_cached_json(cache_key, ranges)

    return ranges

    # mean = df[field].mean()
    # mean_idx = None
    # for i in range(len(bins)):
    #     if bins[1][i] < mean:
    #         mean_idx = i

    # for i, num in enumerate(bins):
    #     state_name = '%s_%s' % (field, i)
    # series.plot.hist(bins=28)
    # plt.show()
