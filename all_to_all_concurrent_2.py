from mpi4py import MPI
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
chunk_size = int(sys.argv[1])
data = bytearray(os.urandom(chunk_size))
s_times = [0] * size
r_times = [0] * size

done = [[True if i == j else False for i in range(size)] for j in range(size)]
ready = False

if rank == 0:
    start = datetime.now()

while not ready:
    occupied = [False for i in range(size)]
    for i in range(size):
        for j in range(size):
            comm.barrier()
            if not (done[i][j] or occupied[i] or occupied[j]):
                if rank == i:
                    s_times[j] = datetime.now()
                    comm.send(data, dest=j, tag=11)
                if rank == j:
                    comm.recv(source=i, tag=11)
                    r_times[i] = datetime.now()
                occupied[i] = True
                occupied[j] = True
                done[i][j] = True
                break

    ready = True
    for i in range(size):
        for j in range(size):
            if not done[i][j]:
                ready = False
                break

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
            if type(time_delta) is not int:
                time_delta = time_delta.total_seconds()
            if time_delta == 0:
                speed = "inf"
            else:
                speed = chunk_size / time_delta
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
    img = img.resize(new_size, resample=Image.NEAREST)
    img.save("concurrent.png")