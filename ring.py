from mpi4py import MPI
import os
import sys
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data_start = bytearray(os.urandom(int(sys.argv[1])))

    time = datetime.now()
    comm.send(data_start, dest=1, tag=11)
    data_end = comm.recv(source=size - 1, tag=11)
    e_time = datetime.now()
    print("finished")
else:
    data = comm.recv(source=rank - 1, tag=11)
    time = datetime.now()
    if rank == size - 1:
        comm.send(data, dest=0, tag=11)
    else:
        comm.send(data, dest=rank+1, tag=11)

time_list = comm.gather(time, root=0)

if rank == 0:
    time_list.append(e_time)
    output = ""
    for i in range(len(time_list) - 1):
        dif = time_list[i + 1] - time_list[i]
        output += "From " + str(i) + " to " + str((i+1) % size) + ": " + str(dif) + "\n"
    print(output)

