import asyncio
import time

async def say_after(delay, what):
    print(delay, f"say_after function started at {time.strftime('%X')}")
    await asyncio.sleep(delay)
    print(what, "delay=", delay, f"say_after function finished at {time.strftime('%X')}")
    return time.strftime('%X')

async def main():

    print(asyncio.get_running_loop())
    # await say_after(2, "two")
    # print("this shows that we wait for the answer to come back with await",
    #     time.strftime('%X'))
    # await say_after(1, "one")
    # print("this shows that we wait for the answer to come back with await",
    #     time.strftime('%X'))

    print("=" * 15, "now, tasks", "=" * 15, time.strftime('%X'))
    print("creating 2 tasks at", time.strftime('%X'))
    task2 = asyncio.create_task(say_after(3, 'world'))
    task1 = asyncio.create_task(say_after(1, 'hello'))
    result = await asyncio.sleep(0,result="about to get status of task 1?")
    print(result)
    print(task1.done())
    main_delay = 1.0
    print("=" * 15, f"sleeping for {main_delay} secs", "="*15, time.strftime('%X'))
    await asyncio.sleep(main_delay)
    print(f"task1 finished?: {task1.done()}")
    print("=" * 15, "and we're back", "=" * 15, time.strftime('%X'))
    print('task 2 finished at', await task2, "but I'm printing at", time.strftime('%X'))
    print("note finish time of task 1:")
    print('task 1 finished at', await task1, "but I'm printing at", time.strftime('%X'))

loop = asyncio.get_event_loop()
print(loop)
asyncio.run(main())
