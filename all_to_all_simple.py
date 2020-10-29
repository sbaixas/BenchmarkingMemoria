from mpi4py import MPI
import os
import sys
from datetime import datetime
import numpy as np
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
chunk_size = int(sys.argv[1])
data = bytearray(os.urandom(chunk_size))
s_times = []
r_times = []

if rank == 0:
    start = datetime.now()

for i in range(size):
    if rank == i:
        for j in range(size):
            if i != j:
                comm.send(data, dest=j, tag=11)
                s_times.append(datetime.now())
            else:
                s_times.append(0)
                r_times.append(0)
    else:
        comm.recv(source=i, tag=11)
        r_times.append(datetime.now())

node_result = [s_times, r_times]

result_data = comm.gather(node_result, root=0)

if rank == 0:
    result = []
    greatest_speed = 0
    lowest_speed = np.inf
    for r in range(len(result_data)):
        result.append([])
        sent_times = result_data[r][0]
        for s in range(len(sent_times)):
            time_delta = result_data[s][1][r] - sent_times[s]
            if r == s:
                speed = "inf"
            else:
                speed = chunk_size / time_delta.total_seconds()
                if speed > greatest_speed:
                    greatest_speed = speed
                if speed < lowest_speed:
                    lowest_speed = speed
            result[r].append(speed)
            print(str(r) + " to " + str(s) + ": " + str(result[r][s]) + " B/s in " + str(time_delta))
    print("Total Processing time: " + str(datetime.now() - start))

    data = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if result[i][j] != 'inf':
                value = int(((result[i][j] - lowest_speed)/(greatest_speed - lowest_speed))*255)
                data[i][j] = [value, value, value]
            else:
                data[i][j] = [0, 0, 255]

    img = Image.fromarray(data)
    new_size = (300, 300)
    img = img.resize(new_size)
    img.show()


