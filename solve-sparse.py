from datetime import datetime
import sys
from mpi4py import MPI
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

mpi_comm = MPI.COMM_WORLD
comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

OptDB = PETSc.Options()
d  = OptDB.getInt('d', int(sys.argv[1]))

A = PETSc.Mat().create(comm=comm)
A.setType('aij')
A.setSizes((d,d))
A.setFromOptions()
A.setPreallocationNNZ((5,5))

Istart,Iend = A.getOwnershipRange()

offset = int(sys.argv[2])

for Ii in range(Istart,Iend):
	v = 4.0
	A.setValue(Ii,Ii,v,addv=True)
	if Ii < d - 1:
		v = -1
		A.setValue(Ii + 1,Ii,v,addv=True)
		A.setValue(Ii,Ii + 1,v,addv=True)


A.assemblyBegin(A.AssemblyType.FINAL)
A.assemblyEnd(A.AssemblyType.FINAL)

A.setOption(A.Option.SYMMETRIC,True)

u = PETSc.Vec().create(comm=comm)
u.setSizes(d)
u.setFromOptions()

b = u.duplicate()
x = b.duplicate()

ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A,A)
rtol=1.e-2/((d+1)*(d+1))
ksp.setTolerances(rtol=rtol,atol=1.e-50)


startTime = datetime.now()

ksp.solve(b,x)

endTime = datetime.now()

startTimes = mpi_comm.gather(startTime, root=0)
endTimes = mpi_comm.gather(endTime, root=0)

if rank == 0:
	startTime = min(startTimes)
	endTime = max(endTimes)
	time_delta = endTime - startTime
	print(time_delta.total_seconds())