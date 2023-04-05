from collections import defaultdict
from ib_insync import tickerTypes, tws_codes as codes
from bot.util import dump_dict

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


if __name__ == "__main__":
    dump_dict(tickerTypes)
    dump_dict(codes.systemMessageCodes, Color="33")
    dump_dict(codes.twsErrorCodes, Color="35")
    dump_dict(codes.clientErrorCodes, Color="36")

