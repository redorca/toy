"""Trading bot.

Usage:
  main.py trade [--type=<contract_type>] [--backtest] [--ignore-market-data] [--limit=<n>] [--obv] [--ema]
  main.py trade_sync [--type=<contract_type>] [--backtest] [--ignore-market-data] [--limit=<n>] [--obv] [--ema]
  main.py cache_warm
  main.py flatten
  main.py tickers
  main.py buy
  main.py buy_sell
  main.py trades
  main.py open_trades
  main.py positions
  main.py cancel_open_trades
  main.py suggest_options
  main.py today_bars
  main.py pull_training <out>
  main.py train <in>
  main.py test
  main.py pnl
  main.py generate_ranges <out> [--days=<n>] [--type=<contract_type>]
  main.py dump_cross_data <out> [--days=<n>] [--type=<contract_type>]
  main.py train_cross_data <in>

Options:
  --backtest              Run in backtest mode.
  --ignore-market-data    Start trading even if market data is not available.
  --type=<contract_type>  One of stocks, fang, options, futures, ufutures, custom. [default: options]
  --limit=<n>             Limit number of contracts to trade on.
  --days=<n>              How many days to analyze. [default: 14]
  --obv                   Use OBV based signal for trading.
  --ema                   Use EMA cross based signal for trading.

"""
import asyncio

from docopt import docopt

import csv
import datetime
import pprint
import math
import time
import sys
import ib_insync as ibs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import joblib
import logging
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from bot.provider import BacktestTickProvider, LiveTickProvider
from bot.analyzer import TechnicalAnalyzer, TrendAnalyzer
from bot.contracts import (OptionsWatchlist,
                           suggest_stocks, suggest_stocks_async,
                           suggest_options, suggest_options_async,
                           suggest_futures, suggest_futures_options,
                           suggest_all_options, suggest_forex,
                           suggest_ranked_options, suggest_micro_futures,
                           suggest_fang_stocks, suggest_crypto,
                           )
from bot.connect import connect, connect_async
from bot.pnl import SignalLogger
from bot.strategy import ToggleStrategy, EmaCrossoverStrategy
from bot.trader import Trader

from bot import conf
from bot import exp
from bot import util
from bot import analyzer
from bot import ranges
from bot import provider
from bot import orders

pp = pprint.PrettyPrinter()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def get_contracts(ib):
    contracts = [
        ibs.Future('ES', '20200918', 'GLOBEX'),
        ibs.Future('NQ', '20200918', 'GLOBEX'),

        # ibs.Stock('NFLX', 'SMART', 'USD'),
        # ibs.Stock('GOOG', 'SMART', 'USD'),
        # ibs.Stock('AAPL', 'SMART', 'USD'),
        # ibs.Stock('TSLA', 'SMART', 'USD'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    return contracts


def get_contracts_of_type(ib, contract_type, limit=100):
    if contract_type == 'options':
        contracts = suggest_options(ib, limit=limit, stocks_limit=50)
    elif contract_type == 'futures':
        contracts = suggest_futures(ib)[:limit]
    elif contract_type == 'ufutures':
        contracts = suggest_micro_futures(ib)[:limit]
    elif contract_type == 'stocks':
        contracts = suggest_stocks(ib)[:limit]
    elif contract_type == 'fang':
        contracts = suggest_fang_stocks(ib)[:limit]
    elif contract_type == 'crypto':
        contracts = suggest_crypto()[:limit]
    elif contract_type == 'custom':
        contracts = [
            ibs.Option('TSLA', '20201023', 445, 'C', 'SMART'),
            ibs.Option('TSLA', '20201023', 445, 'P', 'SMART'),
            ibs.Option('TSLA', '20201023', 450, 'C', 'SMART'),
            ibs.Option('TSLA', '20201023', 450, 'P', 'SMART'),

            ibs.Option('AMZN', '20201023', 3300, 'C', 'SMART'),
            ibs.Option('AMZN', '20201023', 3300, 'P', 'SMART'),
            ibs.Option('AMZN', '20201023', 3400, 'C', 'SMART'),
            ibs.Option('AMZN', '20201023', 3400, 'P', 'SMART'),

            ibs.Option('GOOG', '20201023', 1500, 'C', 'SMART'),
            ibs.Option('GOOG', '20201023', 1500, 'P', 'SMART'),
            ibs.Option('GOOG', '20201023', 1600, 'C', 'SMART'),
            ibs.Option('GOOG', '20201023', 1600, 'P', 'SMART'),
        ]
        # contracts = contracts[:2]
        contracts = [
            ibs.Future('SI', '20201229', 'NYMEX', currency='USD', multiplier=5000),
            # ibs.Stock('AMZN', 'SMART', 'USD'),
            # ibs.Stock('TSLA', 'SMART', 'USD'),
            # ibs.Future('ES', '20201218', 'GLOBEX'),
            # ibs.Future('NQ', '20201218', 'GLOBEX'),
            # ibs.Stock('CCL', 'SMART', 'USD'),
            # ibs.Future('ES', '20201218', 'GLOBEX'),
            # ibs.Option('TSLA', '20201030', 445, 'C', 'SMART'),
            # ibs.Option('TSLA', '20201030', 445, 'P', 'SMART'),
            # ibs.Option('AMZN', '20201030', 3300, 'P', 'SMART'),
            # ibs.Option('AMZN', '20201030', 3400, 'C', 'SMART'),
        ]
    else:
        logging.info('Unexpected contract type', contract_type)
        sys.exit(1)
    return contracts


async def get_contracts_of_type_async(ib, contract_type, limit=100):
    if contract_type == 'options':
        contracts = await suggest_options_async(ib, limit=limit, stocks_limit=50)
    elif contract_type == 'futures':
        contracts = suggest_futures(ib)[:limit]
    elif contract_type == 'ufutures':
        contracts = suggest_micro_futures(ib)[:limit]
    elif contract_type == 'stocks':
        contracts = suggest_stocks(ib)[:limit]
    elif contract_type == 'fang':
        contracts = suggest_fang_stocks(ib)[:limit]
    elif contract_type == 'crypto':
        contracts = suggest_crypto()[:limit]
    elif contract_type == 'custom':
        contracts = [
            ibs.Option('TSLA', '20201023', 445, 'C', 'SMART'),
            ibs.Option('TSLA', '20201023', 445, 'P', 'SMART'),
            ibs.Option('TSLA', '20201023', 450, 'C', 'SMART'),
            ibs.Option('TSLA', '20201023', 450, 'P', 'SMART'),

            ibs.Option('AMZN', '20201023', 3300, 'C', 'SMART'),
            ibs.Option('AMZN', '20201023', 3300, 'P', 'SMART'),
            ibs.Option('AMZN', '20201023', 3400, 'C', 'SMART'),
            ibs.Option('AMZN', '20201023', 3400, 'P', 'SMART'),

            ibs.Option('GOOG', '20201023', 1500, 'C', 'SMART'),
            ibs.Option('GOOG', '20201023', 1500, 'P', 'SMART'),
            ibs.Option('GOOG', '20201023', 1600, 'C', 'SMART'),
            ibs.Option('GOOG', '20201023', 1600, 'P', 'SMART'),
        ]
        # contracts = contracts[:2]
        contracts = [
            ibs.Future('SI', '20201229', 'NYMEX', currency='USD', multiplier=5000),
            # ibs.Stock('AMZN', 'SMART', 'USD'),
            # ibs.Stock('TSLA', 'SMART', 'USD'),
            # ibs.Future('ES', '20201218', 'GLOBEX'),
            # ibs.Future('NQ', '20201218', 'GLOBEX'),
            # ibs.Stock('CCL', 'SMART', 'USD'),
            # ibs.Future('ES', '20201218', 'GLOBEX'),
            # ibs.Option('TSLA', '20201030', 445, 'C', 'SMART'),
            # ibs.Option('TSLA', '20201030', 445, 'P', 'SMART'),
            # ibs.Option('AMZN', '20201030', 3300, 'P', 'SMART'),
            # ibs.Option('AMZN', '20201030', 3400, 'C', 'SMART'),
        ]
    else:
        logging.info('Unexpected contract type', contract_type)
        sys.exit(1)
    return contracts


def run_trading(contract_type="options",
                backtest=False,
                ignore_market_data=True,
                limit=None,
                use_obv=False,
                use_ema=False
                ):
    logging.info('run trading...')
    ib = connect(contract_type=contract_type, client_id=23)

    logging.info('check if mkt data')
    while contract_type == 'options' and not ignore_market_data \
            and not util.is_options_market_data_available(ib):
        logging.info('market data is not available, wait 30 seconds and try again')
        ib.sleep(30)

    # contracts = suggest_futures(ib)[:3]
    # contracts = suggest_stocks(ib)[:3]
    # contracts = suggest_options(ib, limit=15, stocks_limit=10)
    # contracts = suggest_ranked_options(ib, 2, 2)

    if limit is None:
        limit = 30

    contracts = []
    for ct in contract_type.split(','):
        contracts += get_contracts_of_type(ib, ct, limit=int(limit))

    if backtest:
        trader = Trader(ib, conf.ACCOUNT, contracts,
                        bar_size='5 mins',
                        backtest=True,
                        backtest_start=datetime.datetime(year=2020, month=12, day=1,
                                                         hour=1, minute=0, second=0
                                                         ),
                        backtest_duration='10 D',
                        use_obv=use_obv,
                        use_ema=use_ema
                        )
    else:
        trader = Trader(ib, conf.ACCOUNT, contracts, bar_size='5 mins',
                        use_obv=use_obv, use_ema=use_ema
                        )

    trader.run()


async def run_trading_async(contract_type="options",
                            backtest=False,
                            ignore_market_data=True,
                            limit=None,
                            use_obv=False,
                            use_ema=False
                            ):
    logging.warning("run async trading...")

    # ib_task = asyncio.create_task(
    #     connect_async(contract_type=contract_type, client_id=23),
    #     name='connect'
    # )
    # logging.info(type(ib_task), ib_task)
    ib = await connect_async(contract_type=contract_type, client_id=23)
    logging.info('check if mkt data')
    while contract_type == 'options' \
            and not ignore_market_data \
            and not util.is_options_market_data_available(ib):
        logging.info('market data is not available, wait 30 seconds and try again')
        logging.info(type(ib.sleep(30)))
        await ib.sleep(30)

    # contracts = suggest_futures(ib)[:3]
    # contracts = suggest_stocks(ib)[:3]
    # contracts = suggest_options(ib, limit=15, stocks_limit=10)
    # contracts = suggest_ranked_options(ib, 2, 2)

    if limit is None:
        limit = 30

    contracts = []
    for ct in contract_type.split(','):
        contracts += await get_contracts_of_type_async(ib, ct, limit=int(limit))

    if backtest:
        trader = Trader(ib, conf.ACCOUNT, contracts,
                        bar_size='5 mins',
                        backtest=True,
                        backtest_start=datetime.datetime(year=2020, month=12, day=1,
                                                         hour=1, minute=0, second=0
                                                         ),
                        backtest_duration='10 D',
                        use_obv=use_obv,
                        use_ema=use_ema
                        )
    else:
        trader = Trader(ib, conf.ACCOUNT, contracts, bar_size='5 mins',
                        use_obv=use_obv, use_ema=use_ema
                        )

    await trader.run_async()


def cache_warm():
    ib = connect()
    contracts = [
        ibs.Option('TSLA', '20201023', 445, 'C', 'SMART'),
        ibs.Option('TSLA', '20201023', 445, 'P', 'SMART'),
        ibs.Option('TSLA', '20201023', 450, 'C', 'SMART'),
        ibs.Option('TSLA', '20201023', 450, 'P', 'SMART'),
    ]
    # if self.backtest:
    #     provider_class = BacktestTickProvider
    # else:
    #     provider_class = LiveTickProvider

    ib.qualifyContracts(*contracts)
    provider_class = LiveTickProvider
    for i, contract in enumerate(contracts):
        provider_class(ib, contract, bar_size='30 secs', duration='3 D')


def flatten_positions():
    account = conf.ACCOUNT
    ib = connect()
    positions = ib.positions()  # A list of positions, according to IB
    for position in positions:
        # if position.contract.symbol != 'GC':
        #     continue
        if position.account != account:
            logging.info('skip position in account %s' % position.account)
            continue

        logging.info('position', position)
        contract = position.contract
        ib.qualifyContracts(contract)
        totalQuantity = abs(position.position)
        if position.position > 0:  # Number of active Long positions
            action = 'SELL'  # to offset the long positions
        elif position.position < 0:  # Number of active Short positions
            action = 'BUY'  # to offset the short positions
        else:
            assert False

        order = ibs.MarketOrder(action, totalQuantity, account=account, tif='GTC',
                                outsideRth=True)
        logging.info('order', order)
        trade = ib.placeOrder(contract, order)
        logging.info('trade', trade)
        logging.info(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
        assert trade in ib.trades(), 'trade not listed in ib.trades'
    util.slack_message('Flattened positions')


def cancel_open_orders():
    account = conf.ACCOUNT
    ib = connect()
    open_orders = ib.reqAllOpenOrders()
    for open_order in open_orders:
        if open_order.account != account:
            continue
        logging.info('oo', open_order)
        ib.cancelOrder(open_order)


def analyze_trades_csv(filename):
    reader = csv.DictReader(open(filename))
    account = conf.ACCOUNT

    for row in reader:
        if row['Account'] != account:
            continue

        logging.info('Trade: %s %s @ %s %s' % (row['Action'], row['Underlying'], row['Strike'], row['Put/Call']))


def analyze_trades(date_str):
    logging.info('analyze_trades')
    account = conf.ACCOUNT
    ib = connect()

    trades = ib.trades()

    # logging.info(len(trades))
    # return

    for trade in trades:
        # logging.info('trade', trade)

        if trade.order.account != account:
            continue

        fill_dates = [f.time.strftime('%Y-%m-%d') for f in trade.fills]
        if date_str not in fill_dates:
            continue

        logging.info('Trade: %s, %s' % (trade.contract.localSymbol, trade.order.account))

        for fill in trade.fills:
            # import pdb ; pdb.set_trace()
            logging.info('\tFill: %s, %s' % (fill.time, fill.execution.avgPrice))
        # import pdb ; pdb.set_trace()


def pull_training_data(filename):
    ib = connect()
    # ib.reqMarketDataType(3)

    # contract = ibs.Stock('AAPL', 'SMART', 'USD')
    # contract = ibs.Option('FB', '20201009', 260, 'C', 'SMART')
    # contracts = suggest_options(ib, limit=3, stocks_limit=3)
    # logging.info('lets pull training data for %s contracts' % len(contracts))

    # contracts = suggest_options(ib, limit=10, stocks_limit=2,
    #                             limit_strike=2, no_filter=True)
    # logging.info('contracts', contracts)
    # contracts = [ibs.Option('AMZN', '20201016', 3300, 'C', 'SMART')]
    # contracts = [ibs.Stock('AAPL', 'SMART', 'USD')]
    contracts = [
        ibs.Option('AMZN', '20201023', 3100, 'C', 'SMART'),
        ibs.Option('AMZN', '20201023', 3200, 'C', 'SMART'),
        ibs.Option('AMZN', '20201023', 3300, 'C', 'SMART'),
        ibs.Option('AMZN', '20201023', 3400, 'C', 'SMART'),

        ibs.Option('GOOGL', '20201023', 1400, 'C', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1450, 'C', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1500, 'C', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1550, 'C', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1600, 'C', 'SMART'),

        ibs.Option('AMZN', '20201023', 3100, 'P', 'SMART'),
        ibs.Option('AMZN', '20201023', 3300, 'P', 'SMART'),
        ibs.Option('AMZN', '20201023', 3200, 'P', 'SMART'),
        ibs.Option('AMZN', '20201023', 3400, 'P', 'SMART'),

        ibs.Option('GOOGL', '20201023', 1400, 'P', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1450, 'P', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1500, 'P', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1550, 'P', 'SMART'),
        ibs.Option('GOOGL', '20201023', 1600, 'P', 'SMART'),
    ]
    contracts = contracts[:8]

    df = None
    for i, contract in enumerate(contracts):
        part = analyzer.pull_training_data(ib, contract, '2020-10-1', '2020-10-19', read_cache=False)
        # X, Y = analyzer.pull_training_data_for_date(ib, contract, '2020-10-05', read_cache=False)

        logging.info('\n\n\n\t==== %s/%s %s ====\n\n\n' % (i, len(contracts), contract.localSymbol))
        logging.info(contract)
        logging.info(part)
        logging.info(part.describe())

        if df is None:
            df = part
        else:
            df = pd.concat([df, part])

    logging.info(df)
    logging.info(df.describe())
    df.to_pickle(filename)
    logging.info('saved to', filename)

    # import pdb ; pdb.set_trace()
    # idx = (X_full['trailing_volume'] > X_full['trailing_volume'].mean()) & (X_full['adx'] > 40) & (X_full['di+'] > 40)
    # idx = (X_full['trailing_volume'] > X_full['trailing_volume'].mean())
    # idx = (X_full['trailing_volume'] > 100)
    # x_data = X_full[idx]['buy_line']
    # y_data = Y_full[idx]['sum_to_sell_signal']
    # plt.scatter(x_data, y_data)
    # plt.show()


def train(filename):
    df = pd.read_pickle(filename)
    # df = df.loc[df['trailing_volume'] > 100]
    logging.info(df.describe())

    # labels = np.array(df['upcoming_high'])
    labels = np.array(df['upcoming_high_percent'])
    column_list = list(df.columns)
    logging.info('column_list', column_list)

    # feature_list = ['adx', 'obv', 'di+', 'di-', 'cross_signal_int', 'cross_buy_state', 'expiry_days']
    feature_list = [
        'obv',
        'obv_slope',
        'adx_slope',
        'delta_from_mean',
        'obv_buy_line',
        'trailing_volume',
        'adx',
        'cross_buy_state',
        'expiry_days',
        'di+',
        'di-',
        'expiry_days',
    ]
    features = df
    for column_name in column_list:
        if column_name not in feature_list:
            features = features.drop(column_name, axis=1)

    features[features == np.inf] = np.nan
    features.fillna(features.mean(), inplace=True)
    baseline_pred = labels.mean()

    split = int(df.shape[0] * .8)
    train_features, train_labels, test_features, test_labels = features[:split], labels[:split], features[
                                                                                                 split:], labels[split:]
    logging.info('== train ==')
    logging.info(train_features.shape)
    logging.info(train_labels.shape)
    logging.info('== test ==')
    logging.info(test_features.shape)
    logging.info(test_labels.shape)

    baseline_errors = abs(baseline_pred - test_labels)
    logging.info('Average baseline error: ', round(np.mean(baseline_errors), 2))
    clf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # clf = svm.SVR()
    clf.fit(train_features, train_labels)

    predictions = clf.predict(test_features)
    # logging.info('predictions', predictions)
    errors = abs(predictions - test_labels)
    logging.info('Mean Absolute Error:', round(np.mean(errors), 2))
    mse = sklearn.metrics.mean_squared_error(test_labels, predictions)
    # logging.info('mse', mse)

    plt.scatter(predictions, test_labels)
    plt.show()


def print_tickers():
    ib = connect()
    ib.reqMarketDataType(1)
    # contracts = suggest_futures(ib)[:3]
    contracts = [
        ibs.Future('MNQ', '20201218', 'GLOBEX'),
        ibs.Future('MES', '20201218', 'GLOBEX'),
        ibs.Future('M2K', '20201218', 'GLOBEX'),
    ]
    [ib.qualifyContracts(c) for c in contracts]
    # contracts = suggest_forex(ib)
    tick_by_ticks = [ib.reqTickByTickData(c, 'BidAsk') for c in contracts]
    while ib.sleep(1):
        if len(contracts) > 2:
            logging.info('--')
        for tt in tick_by_ticks:
            logging.info('%s\tbid=%s\task=%s\tbidSize=%s\taskSize=%s\tlast=%s' % (
                tt.contract.localSymbol, tt.bid, tt.ask, tt.bidSize, tt.askSize, tt.last))


def place_buy_order():
    ib = connect()
    account = conf.ACCOUNT
    contracts = suggest_forex(ib)
    contract = ibs.Stock('AAPL', 'SMART', 'USD')
    orders.buy(ib, account, contract, 1, backtest=False)


def place_buy_sell_order():
    ib = connect()
    account = conf.ACCOUNT
    contracts = suggest_forex(ib)
    # contract = ibs.Stock('AAPL', 'SMART', 'USD')
    contract = suggest_futures(ib)[0]

    trade = orders.buy(ib, account, contract, 1, num_ticks=2, backtest=False)
    ib.sleep(2)
    logging.info(orders.pretty_string_trade(trade))

    trade = orders.sell(ib, account, contract, 1, num_ticks=2, backtest=False)
    ib.sleep(10)
    logging.info(orders.pretty_string_trade(trade))


def print_trades():
    ib = connect()
    account = conf.ACCOUNT
    orders.pretty_print_trades(ib, account)


def print_open_trades():
    ib = connect()
    account = conf.ACCOUNT
    orders.pretty_print_open_trades(ib, account)


def cancel_open_trades():
    ib = connect()
    account = conf.ACCOUNT
    orders.cancel_open_trades(ib, account)


def print_positions():
    ib = connect()
    account = conf.ACCOUNT
    orders.pretty_print_open_trades(ib, account)
    for p in ib.positions():
        if p.account != account:
            continue
        logging.info(p)


def watch_list_suggest_options():
    ib = connect()
    account = conf.ACCOUNT

    wl = OptionsWatchlist(ib, account)
    for option in wl.candidates():
        logging.info('candidate', option.localSymbol)


def print_today_bars():
    ib = connect()
    contract = ibs.Stock('AAPL', 'SMART', 'USD')
    bars = provider.pull_today_bars(ib, contract, '1 min')


def run_test():
    ib = connect()
    contracts = [
        ibs.Future('ES', '20201218', 'GLOBEX'),
        ibs.Stock('AMZN', 'SMART', 'USD'),
        ibs.Option('AMZN', '20201023', 3200, 'C', 'SMART'),
    ]
    logging.info(util.min_increment(ib, contracts[1]))


def run_pnl():
    ib = connect()
    account = conf.ACCOUNT
    pnl = ib.reqPnL(account)

    pnl_singles = []
    contracts = [p.contract for p in ib.positions(account)]  # without account, fetches all
    # contracts = ib.qualifyContracts(*contracts_) # dumps everything not on NYSE why?
    for contract in contracts:
        foo = ib.reqPnLSingle(account, '', contract.conId)
        pnl_singles.append(foo)
    ib.sleep(2)
    # while True:
    logging.info('=' * 120)

    logging.info(pnl)
    for pnl_single in pnl_singles:
        logging.info(pnl_single)
        # ib.sleep(10)


def run_pnl_as():
    ib = connect()
    account = conf.ACCOUNT
    pnl = ib.reqPnL(account)

    pnl_singles = []
    contracts = [p.contract for p in ib.positions()]
    ib.qualifyContracts(*contracts)
    for contract in contracts:
        pnl_singles.append(ib.reqPnLSingle(account, '', contract.conId))

    while ib.sleep(1):
        logging.info('--')
        logging.info(pnl)
        for pnl_single in pnl_singles:
            logging.info(pnl_single)


def generate_ranges(filename, num_days, contract_type):
    ib = connect()
    contracts = get_contracts_of_type(ib, contract_type)
    logging.info(contracts)
    logging.info('gen ranges', filename, num_days, contracts)
    # for indicator in ['obv', 'adx', 'di+', 'di-']:
    for indicator in ['ema_diff_percent_abs']:
        logging.info('')
        logging.info('====', indicator, '====')
        logging.info('')
        for contract in contracts:
            value_range = ranges.get_ranges(ib, contract, '1 hour', indicator, num_days)
            logging.info('--', contract, '--')
            pp.plogger.info(value_range)


def dump_cross_data(filename, num_days, contract_type):
    ib = connect(contract_type=contract_type, client_id=23)

    contracts = get_contracts_of_type(ib, contract_type)
    end_dt = datetime.datetime.now()
    start_dt = (end_dt - datetime.timedelta(days=num_days))
    end_dt_str = end_dt.strftime('%Y-%m-%d')
    start_dt_str = start_dt.strftime('%Y-%m-%d')

    window_size = 100
    bar_size = '5 mins'

    records = []

    for contract in contracts:
        logging.info('dump cross data', contract)

        df = analyzer.pull_training_data(ib, contract, start_dt_str, end_dt_str, bar_size=bar_size, read_cache=True,
                                         use_rth=False)
        # logging.info(df)
        # import pdb ; pdb.set_trace()

        r = len(df) - 2 * window_size
        for i in range(r):
            if i % 10 == 0:
                logging.info('.', end='', flush=True)
            if i % 500 == 0:
                logging.info('%s/%s (%.1f%%)' % (i, r, float(i * 100) / r))
            idx = i + window_size
            row = df.iloc[idx]

            bar = row.current_bar
            # if bar.datetime.month == 12 and bar.datetime.day == 1 and bar.datetime.hour == 15 and bar.datetime.minute == 0:
            #     import pdb ;pdb.set_trace()

            if row.cross_signal not in ['buy', 'sell']:
                continue

            df_next = df.iloc[idx:idx + window_size]
            df_prev = df.iloc[idx - window_size:idx]
            prev_bars = df_prev.to_dict('records')
            next_bars = df_next.to_dict('records')

            entry_price = row.close
            if row.cross_signal == 'buy':
                peak_key = 'high'
                peak_price = max(df_next['high'])
                dt_peak = min(df[df.high == peak_price]['datetime'])
            else:
                peak_key = 'low'
                peak_price = min(df_next['low'])
                dt_peak = min(df[df.low == peak_price]['datetime'])
            delta = abs(peak_price - entry_price)
            if delta == 0:
                continue

            next_bars_before_peak = df_next[df_next.datetime <= dt_peak]
            bars_before_peak = df_prev[len(next_bars_before_peak):]
            bars_to_peak = pd.concat([bars_before_peak, next_bars_before_peak])

            include_num_bars = 10

            for i in range(len(next_bars_before_peak)):
                item = {}
                item['cross_type'] = row.cross_signal
                for j in range(include_num_bars):
                    record = bars_to_peak.iloc()[len(bars_to_peak) - include_num_bars + j - i]
                    d = record.to_dict()
                    if row.cross_signal == 'buy':
                        percent_increase = (d['high'] - entry_price) / entry_price
                    else:
                        percent_increase = (entry_price - d['low']) / entry_price

                    d['percent_increase'] = percent_increase
                    for key in d.keys():
                        # if 'high' not in key and 'datetime' not in key:
                        #     continue
                        item['from_peak_%s_%s' % (j, key)] = d[key]

                if row.cross_signal == 'buy':
                    percent_of_peak = (d['high'] - entry_price) / delta
                else:
                    percent_of_peak = (entry_price - d['low']) / delta
                item['percent_of_peak'] = percent_of_peak
                item['delta'] = delta / entry_price
                item['is_peak'] = int(d['datetime'] == dt_peak)

                if item['percent_of_peak'] > 1:
                    logging.info('unexpected peak > 1!')
                    import pdb;
                    pdb.set_trace()

                records.append(item)
                # pp.plogger.info(item)
                # if item['is_peak'] and row.cross_signal == 'buy':
                # import pdb ; pdb.set_trace()

    df = pd.DataFrame.from_records(records)
    df.to_pickle(filename)


def train_cross_data(filename):
    df = pd.read_pickle(filename)
    # df = (df[(df.cross_type == 'buy') & (df.delta > .01)]).head(5000)
    df = (df[(df.cross_type == 'buy')]).head(5000)
    df = df.sort_values(['from_peak_0_datetime'])
    logging.info(df.describe())

    labels = np.array(df['percent_of_peak'])
    column_list = list(df.columns)
    # logging.info('column_list', column_list)

    feature_list = []
    for i in range(10):
        # for feature in ['di+', 'adx', 'percent_increase']:
        # for feature in ['percent_increase', 'obv_percent_delta', 'adx', 'di+']:
        for feature in ['di+', 'percent_increase']:
            feature_list.append('from_peak_%s_%s' % (i, feature))
    logging.info('feature_list', feature_list)

    features = df
    for column_name in column_list:
        if column_name not in feature_list:
            features = features.drop(column_name, axis=1)

    features[features == np.inf] = np.nan
    features.fillna(features.mean(), inplace=True)

    split = int(df.shape[0] * .8)
    train_features, train_labels, test_features, test_labels = features[:split], labels[:split], features[
                                                                                                 split:], labels[split:]

    logging.info('== train ==')
    logging.info(train_features.shape)
    logging.info(train_labels.shape)
    logging.info('== test ==')
    logging.info(test_features.shape)
    logging.info(test_labels.shape)

    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=100)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
    reg3 = LinearRegression()

    # clf = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    # clf = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000), max_iter=10000, verbose=1)

    clf = RandomForestRegressor(n_estimators=200, random_state=42)
    clf.fit(train_features, train_labels)
    joblib.dump(clf, 'ema_model.joblib')

    predictions = clf.predict(test_features)
    # logging.info('predictions', predictions)
    errors = abs(predictions - test_labels)
    logging.info('Mean Absolute Error:', round(np.mean(errors), 2))
    mse = sklearn.metrics.mean_squared_error(test_labels, predictions)
    # logging.info('mse', mse)

    # chart_y = test_labels
    logging.info(predictions)
    logging.info(df['percent_of_peak'])
    chart_y = np.array(df['percent_of_peak'])[split:]
    logging.info('max', max(chart_y))
    plt.scatter(predictions, chart_y)
    plt.show()


if __name__ == '__main__':
    logging.critical("docopt for args")
    logging.info("Starting args...")

    args = docopt(__doc__)
    logging.info('data from logs: %s', args)

    if args['trade']:
        asyncio.run(run_trading_async(args['--type'], args['--backtest'], args['--ignore-market-data'], args['--limit'],
                                      args['--obv'],
                                      args['--ema']))
    elif args['trade_sync']:
        run_trading(args['--type'], args['--backtest'], args['--ignore-market-data'], args['--limit'],
                    args['--obv'],
                    args['--ema'])
    elif args['flatten']:
        flatten_positions()
    elif args['cache_warm']:
        cache_warm()
    elif args['tickers']:
        print_tickers()
    elif args['buy']:
        place_buy_order()
    elif args['buy_sell']:
        place_buy_sell_order()
    elif args['trades']:
        print_trades()
    elif args['open_trades']:
        print_open_trades()
    elif args['positions']:
        print_positions()
    elif args['cancel_open_trades']:
        cancel_open_trades()
    elif args['suggest_options']:
        watch_list_suggest_options()
    elif args['today_bars']:
        print_today_bars()
    elif args['pull_training']:
        pull_training_data(args['<out>'])
    elif args['train']:
        train(args['<in>'])
    elif args['test']:
        run_test()
    elif args['pnl']:
        run_pnl()
    elif args['generate_ranges']:
        generate_ranges(args['<out>'], int(args['--days']), args['--type'])
    elif args['dump_cross_data']:
        dump_cross_data(args['<out>'], int(args['--days']), args['--type'])
    elif args['train_cross_data']:
        train_cross_data(args['<in>'])
    else:
        raise Exception('unhandled command')

    logging.fatal("Exiting system...")
    logging.shutdown()

# run_paper_trading()
# flatten_positions()
# cancel_open_orders()
# analyze_trades('2020-09-28')
# analyze_trades_csv('/Users/marcell/Desktop/trades.20200928.csv')
# pull_training_data()
