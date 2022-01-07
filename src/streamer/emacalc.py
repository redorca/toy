from streamer import candles


class EmaCalculator:
    """Arun recommends tulip as math library, replace ta-lib"""

    def __init__(self):
        self.result = "ema result"

    async def run_a(self, candle: candles.Candle):

        return self.result


def konstant(periods):
    k = 2 / (periods + 1)
    knot = 1 - k
    return (k, knot)


def experiment1(time_constant):
    k = konstant(time_constant)
    print("k:", k)
    x = 1
    l = []
    for i in range(1, 3 * time_constant + 1):
        print(i, x)
        x = x * k[1]
        l.append(x)
    print(sum(l[:11]))


# decay constant lambda  is inverse of mean lifetime


def experiment2(time_constant):
    x = 1
    l = []
    for i in np.linspace(1, time_constant * 3, time_constant * 3 + 1):
        y = x * pow(e, -i / time_constant)
        print(i, y)
        l.append(y)
    print(sum(l[:time_constant]))


def ex3(time_constant):
    new_weight, old_weight = konstant(time_constant)
    x = 0
    l = [x]
    for i in range(3 * time_constant + 1):
        print(i, x)

        x = (1 * new_weight) + (x * old_weight)
        l.append(x)
    print(sum(l[:time_constant]))
