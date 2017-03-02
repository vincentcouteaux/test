import os
import time
import sys

def foo(a):
    print("process: {} = {}".format(os.getpid(), a))
    time.sleep(a)
    print("process: {} = FINISHED".format(os.getpid()))

pids = []
for k in [1, 2, 7, 5]:
    for l in range(3):
        pids.append(os.fork())
        if pids[-1] == 0:
            foo(k+l)
            sys.exit(0)
