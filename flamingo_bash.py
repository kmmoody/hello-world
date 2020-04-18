#flamingo
#author: Katherine Moody , contact: kmmo264@g.uky.edu
#last edit: 4/18/2020
#FEniCS fea solver for thermal properties of otter generated RVE models
#Beck Group Computational Materials Science at University of Kentucky
from __future__ import print_function
import os
import sys
import gc
from fenics import *
from mshr import *
from dolfin import *
from ufl import nabla_div
from mpi4py import MPI
#importing model "f" from bash shell
item = sys.argv[1]

#Setting up user classes
#Defining print statements to return singular from multiple threads
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def print_roo(text):
		if rank == 0:
			print(text,flush=True)
		return

#Defining boundary conditions
#All otter generated RVE are cubes of a specified length, and placed 500 from origin
#Each boundary in fenics must be definied by a subdomain, in this case 6 domains for the faces of the "cube"
print_roo('defining boundaries')
tol = 1E-6
#Creating boundaries for boundary conditions

#x500
class BoundaryX500(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] <= 500.0 + tol
#x600
class BoundaryX600(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] >= 600.0 - tol
#y500
class BoundaryY500(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] <= 500.0 + tol
#y600
class BoundaryY600(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] >= 600.0 - tol
#z500
class BoundaryZ500(SubDomain):
	def inside(self, x, on_boundary):
		return x[2] <= 500.0 + tol
#z600
class BoundaryZ600(SubDomain):
	def inside(self, x, on_boundary):
		return x[2] >= 600.0 - tol


print_roo('marking boundaries')

#Allocating variables to store nodes located at the boundary faces
x500 = BoundaryX500()
x600 = BoundaryX600()
y500 = BoundaryY500()
y600 = BoundaryY600()
z500 = BoundaryZ500()
z600 = BoundaryZ600()

#defining gravity and traction for the balancing of equations
f = Constant((0, 0, 0)) 
T = Constant((0, 0, 0))
#Material Properties, k - thermal conductivity
kappa = 0.2
#Function to define heat flux
def hflux(u):
	return -kappa*nabla_grad(u)

#Setting temperature at boundaries
minus_temp = Constant ((0.0, 0.0, 0.0))
plus_temp = Constant((300.0, 300.0, 300.0))

#Direction normals x, y, and z
nx = Constant((1, 0, 0))
ny = Constant((0, 1, 0))
nz = Constant((0, 0, 1))


#Reads the model to create a surface for the structure, and generates a volume mesh based on surface, volume of the mesh is stored for output
surface = Surface3D(path_to_files)
mesh = generate_mesh(surface,10)
volume = assemble(1*dx(mesh))

#Begin boundary conditions loop
#Depending on direction of temperature gradient, boundary conditions are changed accordingly
#Current conditions are set for x, y, z
for ii in range(3):
	gc.collect()
	msg = "Starting calculation {}..."
	print_roo(msg.format(ii)) 

#Defining function space
	V = VectorFunctionSpace(mesh, 'Lagrange', 3)

#Set variable to mark boundaries in the mesh
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() -1)
#setting all the boundaries at value zero so they can be assigned a number
	boundaries.set_all(0)
#Mark the nodes on the boundaries based on the subdomain definitions defined above. Each "cube" face is assigned a number
	x500.mark(boundaries, 1)
	x600.mark(boundaries, 2)
	y500.mark(boundaries, 3)
	y600.mark(boundaries, 4)
	z500.mark(boundaries, 5)
	z600.mark(boundaries, 6)

#Boundary conditions set for each temperature gradient
	print_roo('assigning boundary conditions')

	if ii == 0:
		bcx1 =DirichletBC(V, minus_temp, boundaries, 1)
		bcx2 =DirichletBC(V, plus_temp, boundaries, 2)
		bcx = [bcx1, bcx2]
	elif ii == 1:
		bcy1 =DirichletBC(V, minus_temp, boundaries, 3)
		bcy2 =DirichletBC(V, plus_temp, boundaries, 4)
		bcx = [bcy1, bcy2]
	elif ii == 2:
		bcz1 =DirichletBC(V, minus_temp, boundaries, 5)
		bcz2 =DirichletBC(V, plus_temp, boundaries, 6)
		bcx = [bcz1, bcz2]
	else:
		print_roo('Error selecting boundary conditions!')

	print_roo('setting up equations')
#Setting up variables for function space and solving
	v = TestFunction(V) 
	u_ = TrialFunction(V) 

	#Right and Left hand side of elasticity equation
	a = inner(grad(u_), grad(v))*dx
	L = dot(f, v)*dx + dot(T, v)*ds
	u = Function(V)
	
	print_roo('starting linear solve')

#Setting up solver
#Currently Krylov Solver is implemented to solve by an iterative process with tolerances
	prblm = LinearVariationalProblem(a, L, u, bcx)
	solver = LinearVariationalSolver(prblm)
	MAX_ITERS = 100000
	solver.parameters['linear_solver']= 'gmres'
	solver.parameters['preconditioner']= 'sor'
	solver.parameters['krylov_solver']['monitor_convergence']= True
	solver.parameters['krylov_solver']['absolute_tolerance']=1E-8
	solver.parameters['krylov_solver']['relative_tolerance']=1E-4
	solver.parameters['krylov_solver']['maximum_iterations']=MAX_ITERS
	solver.solve()
	print_roo('finished solving')
#defining heat flux for solution plotting
	heat = hflux(u)
	temp = nabla_grad(u)
	if ii == 0:
		qx = project(dot(dot(heat, nx),nx), FunctionSpace(mesh, 'CG', 1))
		QX = assemble(qx*dx)
		tempgradx = project(dot(dot(temp, nx),nx), FunctionSpace(mesh, 'CG', 1))
		totalt = assemble(tempgradx*dx)
		kxx = -QX/totalt
		qy = project(dot(dot(heat, ny),ny), FunctionSpace(mesh, 'CG', 1))
		QY = assemble(qy*dx)
		kxy = -QY/totalt
		qz = project(dot(dot(heat, nz),nz), FunctionSpace(mesh, 'CG', 1))
		QZ = assemble(qz*dx)
		kxz = -QZ/totalt
	elif ii == 1:
		qx = project(dot(dot(heat, nx),nx), FunctionSpace(mesh, 'CG', 1))
		QX = assemble(qx*dx)
		tempgrady = project(dot(dot(temp, ny),ny), FunctionSpace(mesh, 'CG', 1))
		totalt = assemble(tempgrady*dx)
		kyx = -QX/totalt
		qy = project(dot(dot(heat, ny),ny), FunctionSpace(mesh, 'CG', 1))
		QY = assemble(qy*dx)
		kyy = -QY/totalt
		qz = project(dot(dot(heat, nz),nz), FunctionSpace(mesh, 'CG', 1))
		QZ = assemble(qz*dx)
		kyz = -QZ/totalt
	elif ii == 2:
		qx = project(dot(dot(heat, nx),nx), FunctionSpace(mesh, 'CG', 1))
		QX = assemble(qx*dx)
		tempgradz = project(dot(dot(temp, nz),nz), FunctionSpace(mesh, 'CG', 1))
		totalt = assemble(tempgradz*dx)
		kzx = -QX/totalt
		qy = project(dot(dot(heat, ny),ny), FunctionSpace(mesh, 'CG', 1))
		QY = assemble(qy*dx)
		kzy = -QY/totalt
		qz = project(dot(dot(heat, nz),nz), FunctionSpace(mesh, 'CG', 1))
		QZ = assemble(qz*dx)
		kzz = -QZ/totalt
	else:
		print_roo('Error calculating thermal conductivity')

	print_roo('deleting variables...')
	del V
	del boundaries
	del bcx
	del u_
	del u
	del v
	del a
	del L
	del prblm
	del solver

output=open(item, 'w')
output.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" %("structure =", item, "volume=", volume, "kxx=", kxx, "kxy=", kxy, "kxz=", kxz, "kyx=", kyx, "kyy=", kyy, "kyz=", kyz, "kzx=", kzx, "kzy=", kzy, "kzz=", kzz))
out.close()


del mesh
del surface
del heat
del temp
del tempgradz
del tempgrady
del tempgradx
del kxx
del kxy
del kxz
del kyx
del kyy
del kyz
del kzx
del kzy
del kzz

print_roo('At end')
