import os
import time
import sys

def foo(a):
    print("process: {} = {}".format(os.getpid(), a))
    time.sleep(a)
    print("process: {} = FINISHED".format(os.getpid()))
    return a

pids = []
f = open("test.txt", "w")
for k in [1, 2, 7, 5]:
    for l in range(3):
        pids.append(os.fork())
        if pids[-1] == 0:
            f.write("aaaaa"+str(foo(k+l)))
            sys.exit(0)
f.close()
