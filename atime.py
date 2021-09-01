import asyncio
import time

from socket import gethostname


def elapsed_time(start:float):
    now_ = time.perf_counter
    return now_ - start


def timeit(func):
    async def process(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            print(func.__name__,'entering func  a coroutine: {}'.format(func.__name__),' w args ', func.__code__.co_varnames, ' no of args', func.__code__.co_argcount, ' defaults', func.__defaults__)#🚀
            # loop = asyncio.get_event_loop()
            # print(loop)
            return await func(*args, **params)
        else:
            print('this is not a coroutine: {}'.format(func.__name__),' w args ', func.__code__.co_varnames, ' no of args', func.__code__.co_argcount, ' defaults', func.__defaults__)

            return func(*args, **params)

    async def helper(*args, **params):
        print('{}.time'.format(func.__name__))
        start = time.time()
        result = await process(func, *args, **params)

        # Test normal function route...
        # result = await process(lambda *a, **p: print(*a, **p), *args, **params)
        print('host:',gethostname(),end='')
        print('>>> ',func, time.time() - start, ' seconds ', func,args,params,func,' took ', '%.4f'%(time.time() - start),' seconds ⌛ to run ',func,' w args ', len(args), ' w params ', len(params) ) #⏰
        return result

    return helper
timeit_async = timeit_async_coro = timeit_coro = timeit_async_coro_decorator=timeit

async def compute(x, y):
    print('Compute %s + %s ...' % (x, y))
    await asyncio.sleep(1.0)  # asyncio.sleep is also a coroutine
    return x + y


@timeit
async def print_sum(x, y):
    result = await compute(x, y)
    print('%s + %s = %s' % (x, y, result))

if __name__ == '__main__'and "get_ipython" not in dir():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(print_sum(1, 2))
    loop.close()
