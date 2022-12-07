import ib_insync as ibs

from decimal import Decimal

from bot import util


def pretty_string_trade(trade):
    try:
        timestamp = trade.log[0].time.strftime('%Y-%d-%m %H:%M:%S')
    except IndexError:
        timestamp = 'no timestamp    '

    return '%s\t%s\t%s\t%s @ $%s\t[%s]\t%s' % (timestamp, trade.order.action, trade.contract.localSymbol, trade.order.totalQuantity, trade.order.lmtPrice, ','.join([l.status for l in trade.log]), trade.order.__class__.__name__)


def pretty_print_trades(ib, account):
    trades = ib.trades()
    for trade in trades:
        if trade.order.account != account:
            continue
        print('- %s' % pretty_string_trade(trade))


def pretty_print_open_trades(ib, account):
    trades = ib.openTrades()
    for trade in trades:
        if trade.order.account != account:
            continue
        print('- %s' % pretty_string_trade(trade))


def cancel_open_trades(ib, account):
    trades = ib.openTrades()
    for trade in trades:
        if trade.order.account != account:
            continue
        ib.cancelOrder(trade.order)
        print('Cancel: %s' % (pretty_string_trade(trade)))


def open_trades_for_contract(ib, contract):
    ib.qualifyContracts(contract)
    trades = ib.openTrades()
    result = []
    for trade in trades:
        if trade.contract.conId == contract.conId:
            result.append(trade)
    return result


def buy(ib, account, contract, quantity, backtest=False, num_ticks=2):
    if backtest:
        print('Backtest, NOT placing BUY order', quantity, contract)
        return

    ib.qualifyContracts(contract)
    details = ib.reqContractDetails(contract)
    [ticker] = ib.reqTickers(contract)
    ask = None
    while ask is None and ib.sleep(.01):
        ask = ticker.ask
        print('ask', ask)

    limit_price = ask + num_ticks * details[0].minTick
    # order = ibs.MarketOrder('BUY', quantity, account=account, tif='GTC')
    order = ibs.LimitOrder('BUY', quantity, '%s' % limit_price, account=account, tif='GTC')

    trade = ib.placeOrder(contract, order)
    ib.sleep(.1)
    print('\n\t%s\n' % pretty_string_trade(trade))
    util.slack_message('Sent BUY: %s' % pretty_string_trade(trade))
    return trade


def sell(ib, account, contract, quantity, backtest=False, num_ticks=2):
    if backtest:
        print('Backtest, NOT placing SELL order', quantity, contract)
        return

    ib.qualifyContracts(contract)
    details = ib.reqContractDetails(contract)
    [ticker] = ib.reqTickers(contract)
    bid = None
    while bid is None and ib.sleep(.01):
        bid = ticker.bid
        print('bid', bid)

    limit_price = bid - num_ticks * details[0].minTick
    # order = ibs.MarketOrder('BUY', quantity, account=account, tif='GTC')
    order = ibs.LimitOrder('SELL', quantity, '%s' % limit_price, account=account, tif='GTC')

    trade = ib.placeOrder(contract, order)
    ib.sleep(.1)
    print('\n\t%s\n' % pretty_string_trade(trade))
    util.slack_message('Sent SELL: %s' % pretty_string_trade(trade))
    return trade
