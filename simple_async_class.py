"""
The asynchronous constructor trick
https://stackoverflow.com/questions/49674514/start-async-task-in-class
"""

import asyncio

class Test:

    async def hello_world(self):
        counter = 10
        while counter > 0:
            print("Hello World!")
            counter -= 1
            await asyncio.sleep(1)
        return self

    def __await__(self):
        # hello_world() returns a coro, which is then awaited.
        # presumes, I think, a running loop and probably fails without one.
        return self.hello_world().__await__()


async def main():
    test = await Test()


asyncio.run(main())

# but this seems lame, in the sense that it doesn't do anything interesting
# instead, how about
# https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init?noredirect=1&lq=1

# create an async class method to create the object
import asyncio

dsn = "..."


class Foo(object):
    @classmethod
    async def create(cls, settings):
        self = Foo()
        self.settings = settings
        self.pool = await create_pool(dsn)
        return self


async def main(settings):
    settings = "..."
    foo = await Foo.create(settings)

# or just an ordinary function
import asyncio

dsn = "..."


async def create_foo(settings):
    foo = Foo(settings)
    await foo._init()
    return foo


class Foo(object):
    def __init__(self, settings):
        self.settings = settings

    async def _init(self):
        self.pool = await create_pool(dsn)


async def main():
    settings = "..."
    foo = await create_foo(settings)

#  I'm unimpressed seems ugly and non-Pythonic

import asyncio

class Foo:
    def __init__(self, settings):
        self.settings = settings

    async def async_init(self):
        await create_pool(dsn)

    def __await__(self):
        return self.async_init().__await__()

loop = asyncio.get_event_loop()
foo = loop.run_until_complete(Foo(settings))

# Basically what happens here is __init__() gets called first as usual.
# Then __await__() gets called which then awaits async_init().
