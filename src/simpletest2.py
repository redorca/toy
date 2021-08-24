import asyncio
import time

async def echo(i:int=41):
    await asyncio.sleep(10 + 3 * i)
    return i + 1

async def simple_sleeper(count:int):
    "async sleep for 1 second, and print the count"
    for i in range(count):
        await asyncio.sleep(1)
        print(f"that was {i+1} of {count}")

async def tasked_sleeper(count):
    """launch 'count' tasks sleeping 1 to count seconds"""
    async def t_s(seconds):
        await asyncio.sleep(seconds)
        print (f"that was {seconds} seconds")
        return seconds

    tasks = set()
    for i in range(1,count+1):
        tasks.add(asyncio.create_task(t_s(i), name=f"task {i}"))
    # done, pending = await asyncio.wait(tasks, return_when='FIRST_COMPLETED')
    for coro in asyncio.as_completed(tasks):
        foo = await coro
        print(f"that was coro {foo}")
    print("done")


async def main():
    start = time.perf_counter()
    tasks = []
    for i in range(50):
        tasks.append(echo(i))

    results = await asyncio.gather(*tasks)
    finish = time.perf_counter()
    print (results)
    print(finish - start)

async def main_tasks():
    how_many = 5
    start = time.perf_counter()
    real_tasks = []
    for i in range(how_many):
        real_tasks.append(asyncio.create_task(echo(i),name=f"task{i}"))
        if i == 1:
            await real_tasks[0]
    # await asyncio.sleep(0)
    # results = await asyncio.gather(*real_tasks)
    # await real_tasks[how_many - 1]
    while len(real_tasks) > 0:
        await asyncio.sleep(0)
        task = real_tasks.pop(0)
        if task.done():
            print (task.get_name(), task.result())
            # not_done -= 1
        else:
            if len(real_tasks) == 2: await task
            real_tasks.append(task)

    finish = time.perf_counter()
    # print(results)
    print(finish - start)


# asyncio.run(main_tasks())

asyncio.run(tasked_sleeper(7))
