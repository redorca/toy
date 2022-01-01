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
