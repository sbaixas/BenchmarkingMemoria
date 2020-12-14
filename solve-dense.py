from datetime import datetime
import sys
from mpi4py import MPI
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
A.setType('dense')
A.setSizes((d,d))
A.setFromOptions()

A.setUp()	


Istart,Iend = A.getOwnershipRange()

for Ii in range(Istart, Iend):
	for Ij in range(0, d):
		if Ij != Ii:
			v = Ij - Ii
		else:
			v = Ij
		A.setValue(Ii,Ij,v,addv=True)

A.assemblyBegin(A.AssemblyType.FINAL)
A.assemblyEnd(A.AssemblyType.FINAL)


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