from collections import defaultdict

# more tickTypes at https://interactivebrokers.github.io/tws-api/tick_types.html
type2name = defaultdict(
    lambda: "unregistered tickType",
    {
        0: "lots@bid price",
        1: "high bid",
        2: "low bid",
        4: "last price",
        5: "lots@last price",
        6: "daily high",
        7: "daily low",
        8: "daily volume * 100",
        9: "close price",
        14: "open price",
    },
)

tickTypes = [
        ["Bid Size",   "Number of contracts or lots offered at the bid price."],
        ["Bid Price",   "Highest priced bid for the contract."],
        ["Ask Price",   "Lowest price offer on the contract."],
        ["Ask Size",   "Number of contracts or lots offered at the ask price."],
        ["Last Price",   "Last price at which the contract traded (does not include some trades in RTVolume)."],
        ["Last Size",   "Number of contracts or lots traded at the last price."],
        ["High",   "High price for the day."],
        ["Low",   "Low price for the day."],
        ["Volume",   "Trading volume for the day for the selected contract (US Stocks: multiplier 100)."],
        ["Close Price",   "The last available closing price for the <em>previous</em>"],
        ["Bid Option Computation",   "Computed Greeks and implied volatility based on the underlying stock price and the option bid price. "],
        ["Ask Option Computation",   "Computed Greeks and implied volatility based on the underlying stock price and the option ask price. "],
        ["Last Option Computation",   "Computed Greeks and implied volatility based on the underlying stock price and the option last traded price. "],
        ["Model Option Computation",   "Computed Greeks and"],
        ["Open Tick",   "Current session's opening price. Before"],
        ["Low 13 Weeks",   "Lowest price for the last 13 weeks. For stocks only."],
        ["High 13 Weeks",   "Highest price for the last 13 weeks. For stocks only."],
        ["Low 26 Weeks",   "Lowest price for the last 26 weeks. For stocks only."],
        ["High 26 Weeks",   "Highest price for the last 26 weeks. For stocks only."],
        ["Low 52 Weeks",   "Lowest price for the last 52 weeks. For stocks only."],
        ["High 52 Weeks",   "Highest price for the last 52 weeks. For stocks only."],
        ["Average Volume",   "The average daily trading volume over 90 days. Multiplier of 100. For stocks only."],
        ["Open Interest",   "(Deprecated"],
        ["Option Historical Volatility",   "The 30-day historical volatility (currently for stocks)."],
        ["Option Implied Volatility",   "A prediction of how"],
        ["Option Bid Exchange",   "Not Used."],
        ["Option Ask Exchange",   "Not Used."],
        ["Option Call Open Interest",   "Call option open interest."],
        ["Option Put Open Interest",   "Put option open interest."],
        ["Option Call Volume",   "Call option volume for the trading day."],
        ["Option Put Volume",   "Put option volume for the trading day."],
        ["Index Future Premium",   "The number of points that the index is over the cash index."],
        ["Bid Exchange",   "For stock and options"],
        ["Ask Exchange",   "For stock and options"],
        ["Auction Volume",   "The number of shares that would trade if no new orders were received and the auction were held now."],
        ["Auction Price",   "The price at which the auction"],
        ["Auction Imbalance",   "The number of unmatched shares"],
        ["Mark Price",   "The mark price is the current"],
        ["Bid EFP Computation",   "Computed EFP bid price"],
        ["Ask EFP Computation",   "Computed EFP ask price"],
        ["Last EFP Computation",   "Computed EFP last price"],
        ["Open EFP Computation",   "Computed EFP open price"],
        ["High EFP Computation",   "Computed high EFP traded price for the day"],
        ["Low EFP Computation",   "Computed low EFP traded price for the day"],
        ["Close EFP Computation",   "Computed closing EFP price for previous day"],
        ["Last Timestamp",   "Time of the last trade (in UNIX time)."],
        ["Shortable",   "Describes the level of difficulty with which the contract can be sold short. "],
        ["RT Volume (Time &amp; Sales)",   "Last trade details (Including both \"Last\" and \"Unreportable Last\" trades). "],
        ["Halted",   "Indicates if a contract is halted. "],
        ["Bid Yield",   "Implied yield of the bond if it is purchased at the current bid."],
        ["Ask Yield",   "Implied yield of the bond if it is purchased at the current ask."],
        ["Last Yield",   "Implied yield of the bond if it is purchased at the last price."],
        ["Custom Option Computation",   "Greek values are based off a user customized price."],
        ["Trade Count",   "Trade count for the day."],
        ["Trade Rate",   "Trade count per minute."],
        ["Volume Rate",   "Volume per minute."],
        ["Last RTH Trade",   "Last Regular Trading Hours traded price."],
        ["RT Historical Volatility",   "30-day real time historical volatility."],
        ["IB Dividends",   "Contract's dividends. "],
        ["Bond Factor Multiplier",   "The bond factor is a number that indicates the ratio of the current bond principal to the original principal"],
        ["Regulatory Imbalance",   "The imbalance that is used"],
        ["News",   "Contract's news feed."],
        ["Short-Term Volume 3 Minutes",   "The past three minutes volume. Interpolation may be applied. For stocks only."],
        ["Short-Term Volume 5 Minutes",   "The past five minutes volume. Interpolation may be applied. For stocks only."],
        ["Short-Term Volume 10 Minutes",   "The past ten minutes volume. Interpolation may be applied. For stocks only."],
        ["Delayed Bid",   "Delayed bid price. "],
        ["Delayed Ask",   "Delayed ask price. "],
        ["Delayed Last",   "Delayed last traded price. "],
        ["Delayed Bid Size",   "Delayed bid size. "],
        ["Delayed Ask Size",   "Delayed ask size. "],
        ["Delayed Last Size",   "Delayed last size. "],
        ["Delayed High Price",   "Delayed highest price of the day. "],
        ["Delayed Low Price",   "Delayed lowest price of the day. "],
        ["Delayed Volume",   "Delayed traded volume of the day. "],
        ["Delayed Close",   "The <b>prior</b> day's closing price."],
        ["Delayed Open",   "Not currently available"],
        ["RT Trade Volume",   "Last trade details that excludes \"Unreportable Trades\". "],
        ["Creditman mark price",   "Not currently available"],
        ["Creditman slow mark price",   "Slower mark price update used in system calculations"],
        ["Delayed Bid Option",   "Computed greeks based on delayed bid price. "],
        ["Delayed Ask Option",   "Computed greeks based on delayed ask price. "],
        ["Delayed Last Option",   "Computed greeks based on delayed last price. "],
        ["Delayed Model Option",   "Computed Greeks and model's implied volatility based on delayed stock and option prices."],
        ["Last Exchange",   "Exchange of last traded price"],
        ["Last Regulatory Time",   "Timestamp (in Unix ms time) of last trade returned with regulatory snapshot"],
        ["Futures Open Interest",   "Total number of outstanding futures contracts (TWS v965+). *HSI open interest requested with generic tick 101"],
        ["Average Option Volume",   "Average volume of the corresponding option contracts(TWS Build 970+ is required)"],
        ["Delayed Last Timestamp",   "Delayed time of the last trade (in UNIX time) (TWS Build 970+ is required)"],
        ["Shortable Shares",   "Number of shares available to short (TWS Build 974+ is required)"],
        ["ETF Nav Close",   "Today's closing price of ETF's Net Asset Value (NAV). Calculation is based on prices of ETF's underlying securities."],
        ["ETF Nav Prior Close",   "Yesterday's closing price of ETF's Net Asset Value (NAV). Calculation is based on prices of ETF's underlying securities."],
        ["ETF Nav Bid",   "The bid price of ETF's Net Asset Value (NAV). Calculation is based on prices of ETF's underlying securities."],
        ["ETF Nav Ask",   "The ask price of ETF's Net Asset Value (NAV). Calculation is based on prices of ETF's underlying securities."],
        ["ETF Nav Last",   "The last price of Net Asset Value"],
        ["ETF Nav Frozen Last",   "ETF Nav Last for Frozen data"],
        ["ETF Nav High",   "The high price of ETF's Net Asset Value (NAV)"],
        ["ETF Nav Low",   "The low price of ETF's Net Asset Value (NAV)"],
        ["Estimated IPO - Midpoint",   "Midpoint is calculated based on IPO price range"],
        ["Final IPO Price",   "Final price for IPO"],
    ]

def dump_tick_types():
    '''
    list each tick code, name, and description
    '''
    for idx in range(0, len(tickTypes)):
        print(f'        {idx} : \"{tickTypes[idx][0]}\" \"{tickTypes[idx][1]}\"')

if __name__ == "__main__":
    dump_tick_types()

