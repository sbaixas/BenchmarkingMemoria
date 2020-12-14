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


def generate_ranges(n):
    result = []
    rounded = int(size/n)
    if rounded > 1:
        for i in range(n):
            if i == n-1:
                result.append(range(i * rounded, size))
            else:
                result.append(range(i*rounded, (i+1)*rounded))
        return result
    return range(size)


ranges = generate_ranges(2)
if rank == 0:
    start = datetime.now()
for range_i in ranges:
    if rank in range_i:
        for i in range_i:
            if rank == i:
                for j in range_i:
                    if i != j:
                        s_times[j] = datetime.now()
                        comm.send(data, dest=j, tag=11)
                    else:
                        s_times[i] = 0
                        r_times[j] = 0
            else:
                comm.recv(source=i, tag=11)
                r_times[i] = datetime.now()


for range_i in ranges:

    if rank in ranges[0]:
        for i in ranges[0]:
            if rank == i:
                for j in ranges[1]:
                    if i != j:
                        s_times[j] = datetime.now()
                        comm.send(data, dest=j, tag=11)
                    else:
                        s_times[i] = 0
                        r_times[j] = 0
        for i in ranges[1]:
            comm.recv(source=i, tag=11)
            r_times[i] = datetime.now()

    if rank in ranges[1]:
        for i in ranges[0]:
            comm.recv(source=i, tag=11)
            r_times[i] = datetime.now()
        for i in ranges[1]:
            if rank == i:
                for j in ranges[0]:
                    if i != j:
                        s_times[j] = datetime.now()
                        comm.send(data, dest=j, tag=11)
                    else:
                        s_times[i] = 0
                        r_times[j] = 0

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




