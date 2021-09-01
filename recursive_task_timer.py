import asyncio
import datetime, time

async def display_date(end_time, recursion):
    now = time.perf_counter()
    tasks = []
    loop = asyncio.get_running_loop()
    if now < end_time:
        depth = recursion + 1
        await asyncio.sleep(1)
        print(f"at level {recursion}, creating task {depth}")
        task =  asyncio.create_task(
            display_date2( end_time, depth  ) , name=f"rec{depth:3d}"
        )
        await task
        print(f"exiting level {recursion}")


async def display_date2(end_time, recursion):
    task = asyncio.create_task(display_date(end_time, recursion), name=f"rec{recursion:06d}")
    # await task
    sleepfor = 1 #  it's only the last one that matters
    print(f'recursion {recursion} about to sleep for {sleepfor} second')
    await asyncio.sleep(sleepfor)
    await task # task starts executing at create_create task. await syncs answer
    # move await task to just under task create to de-parallel sleep and display_date task



end_time = time.perf_counter() + 10.1
startat = time.perf_counter()
asyncio.run(display_date(end_time, 0), debug=True)
print(f"that took {time.perf_counter() - startat} secs")
