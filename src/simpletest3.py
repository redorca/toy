"""one way to iterate through a task list."""
import asyncio
import time


async def nada(x,start):
    print(f"{time.perf_counter()-start}:in nada, x was {x}")
    return x


async def simple_sleeper(count, start):
    "async sleep for 1 second, and print the count"
    print(f"sleeper started at {time.perf_counter()-start}")
    for i in range(count):
        await asyncio.sleep(1)
        print(f"{time.perf_counter()-start}:that was {i+1} of {count}")
    return f"sleeper done"



async def main():
    start = time.perf_counter()

    task1 = asyncio.create_task(nada(1,start), name='nada1')
    task2 = asyncio.create_task(nada(2, start), name='nada2')
    task3 = asyncio.create_task(nada(3, start), name='nada3')
    task4 = asyncio.create_task(simple_sleeper(5, start), name="sleeper")

    tasks = [t for t in (task4, task1, task2, task3)]

    for x in asyncio.as_completed(tasks):
        y = await x
        print(type(y), y)
        if y == 2:
            await asyncio.sleep(3)

    print(f"total time was {time.perf_counter()- start}")


async def main_tasks():
    how_many = 5
    start = time.perf_counter()
    real_tasks = []
    for i in range(how_many):
        real_tasks.append(asyncio.create_task(echo(i), name=f"task{i}"))
        if i == 1:
            await real_tasks[0]
    # await asyncio.sleep(0)
    # results = await asyncio.gather(*real_tasks)
    # await real_tasks[how_many - 1]
    while len(real_tasks) > 0:
        await asyncio.sleep(0)
        task = real_tasks.pop(0)
        if task.done():
            print(task.get_name(), task.result())
            # not_done -= 1
        else:
            if len(real_tasks) == 2:
                await task
            real_tasks.append(task)

    finish = time.perf_counter()
    # print(results)
    print(finish - start)


# asyncio.run(main_tasks())

asyncio.run(main())
