#kangaroo
#author: Katherine Moody , contact: kmmo264@g.uky.edu
#last edit: 4/18/2020
#FEniCS fea solver for mechanical properites of otter generated RVE models
#Beck Group Computational Materials Science at University of Kentucky
#PI Dr. Matthew J Beck m.beck@uky.edu
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

print('WELCOME TO KANGAROOF!!')
#Setting up user classes
#Defining print statements to return singular from multiple threads
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def print_roo(text):
    if rank == 0:
        print(text,flush=True)
    return
print_roo(item)

#Defining boundary conditions
#All otter generated RVE are cubes of a specified length, and placed 500 from origin
#Each boundary in fenics must be definied by a subdomain, in this case 6 domains for the faces of the "cube"
print_roo('defining boundaries')
tol = 1E-6

class BoundaryX500(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] <= 500.0 + tol
class BoundaryX600(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] >= 600.0 - tol
class BoundaryY500(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] <= 500.0 + tol
class BoundaryY600(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] >= 600.0 - tol
class BoundaryZ500(SubDomain):
	def inside(self, x, on_boundary):
		return x[2] <= 500.0 + tol
class BoundaryZ600(SubDomain):
	def inside(self, x, on_boundary):
		return x[2] >= 600.0 - tol

#Allocating variables to store nodes located at the boundary faces
print_roo('marking boundaries')
x500 = BoundaryX500()
x600 = BoundaryX600()
y500 = BoundaryY500()
y600 = BoundaryY600()
z500 = BoundaryZ500()
z600 = BoundaryZ600()

#defining gravity and traction for the balancing of equations
f = Constant((0, 0, 0))
T = Constant((0, 0, 0))

#Defining material properties
#E- elastic modules (Pascals), nu - Poisson's ratio
E = 69e9 
nu = 0.33 
#Lame properties
mu =Constant(E/2/(1+nu))
lmbda =Constant(E*nu/((1+nu)*(1-2*nu)))

#Direction normals x, y, and z
nx = Constant((1, 0, 0))
ny = Constant((0, 1, 0))
nz = Constant((0, 0, 1))

#Strain
def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
#Stress
def sigma(u):
    return lmbda*nabla_div(u)*Identity(3) + 2*mu*eps(u)
#Strain energy
def en_dens(u):
    str_ele = eps(u)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC


#Reads the model to create a surface for the structure, and generates a volume mesh based on surface, volume of the mesh is stored for output
surface = Surface3D(item)
mesh = generate_mesh(surface,0.5)
volume = assemble(1*dx(mesh))

#Begin boundary conditions loop
#Depending on direction of displacement, boundary conditions are changed accordingly
#Current conditions are set for x, y, z, and hydrostatic displacements of 1%
for ii in range(4):
	gc.collect()
	msg = "Starting calculation {}..."
	print_roo(msg.format(ii)) 

#Function space for calculations defined
	V = VectorFunctionSpace(mesh, 'Lagrange', 3)

#Set variable to mark boundaries in the mesh
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() -1)
	boundaries.set_all(0)
#Mark the nodes on the boundaries based on the subdomain definitions defined above. Each "cube" face is assigned a number
	x500.mark(boundaries, 1)
	x600.mark(boundaries, 2)
	y500.mark(boundaries, 3)
	y600.mark(boundaries, 4)
	z500.mark(boundaries, 5)
	z600.mark(boundaries, 6)

#Boundary conditions set for each displacement direction
	print_roo('assigning boundary conditions')

	if ii == 0:
		bcx1 =DirichletBC(V, Constant((0.0,0.0,0.0)), boundaries, 1)
		bcx2 =DirichletBC(V.sub(0), Constant(-1.0), boundaries, 2)
		bcy1 =DirichletBC(V.sub(1), Constant(0.0), boundaries, 3)
		bcy2 =DirichletBC(V.sub(1), Constant(0.0), boundaries, 4)
		bcz1 =DirichletBC(V.sub(2), Constant(0.0), boundaries, 5)
		bcz2 =DirichletBC(V.sub(2), Constant(0.0), boundaries, 6)
		bcx = [bcx1, bcx2, bcy1, bcy2, bcz1, bcz2]
	elif ii == 1:
		bcx1 =DirichletBC(V.sub(0), Constant(0.0), boundaries, 1)
		bcx2 =DirichletBC(V.sub(0), Constant(0.0), boundaries, 2)
		bcy1 =DirichletBC(V, Constant((0.0,0.0,0.0)), boundaries, 3)
		bcy2 =DirichletBC(V.sub(1), Constant(-1.0), boundaries, 4)
		bcz1 =DirichletBC(V.sub(2), Constant(0.0), boundaries, 5)
		bcz2 =DirichletBC(V.sub(2), Constant(0.0), boundaries, 6)
		bcx = [bcx1, bcx2, bcy1, bcy2, bcz1, bcz2]
	elif ii == 2:
		bcx1 =DirichletBC(V.sub(0), Constant(0.0), boundaries, 1)
		bcx2 =DirichletBC(V.sub(0), Constant(0.0), boundaries, 2)
		bcy1 =DirichletBC(V.sub(1), Constant(0.0), boundaries, 3)
		bcy2 =DirichletBC(V.sub(1), Constant(0.0), boundaries, 4)
		bcz1 =DirichletBC(V, Constant((0.0,0.0,0.0)), boundaries, 5)
		bcz2 =DirichletBC(V.sub(2), Constant(-1.0), boundaries, 6)
		bcx = [bcx1, bcx2, bcy1, bcy2, bcz1, bcz2]
	elif ii == 3:
		bcx1 =DirichletBC(V.sub(0), Constant(0.0), boundaries, 1)
		bcx2 =DirichletBC(V.sub(0), Constant(-1.0), boundaries, 2)
		bcy1 =DirichletBC(V.sub(1), Constant(0.0), boundaries, 3)	
		bcy2 =DirichletBC(V.sub(1), Constant(-1.0), boundaries, 4)
		bcz1 =DirichletBC(V.sub(2), Constant(0.0), boundaries, 5)
		bcz2 =DirichletBC(V.sub(2), Constant(-1.0), boundaries, 6)
		bcx = [bcx1, bcx2, bcy1, bcy2, bcz1, bcz2]
	else:
		print_roo('Error selecting boundary conditions!')

#Setting up variables for function space and solving
	print_roo('Setting up equations')
	v = TestFunction(V) 
	u_ = TrialFunction(V)

#Right and Left hand side of elasticity equation
	a = inner(sigma(u_), eps(v))*dx
	L = dot(f, v)*dx + dot(T, v)*ds

	u = Function(V)
	print_roo('starting linear solve')

#Setting up solver
#Currently Krylov Solver is implemented to solve by an iterative process with tolerances
	prblm = LinearVariationalProblem(a, L, u, bcx)
	solver = LinearVariationalSolver(prblm)
	MAX_ITERS = 6000
			#parameters.linear_algebra_backend
	solver.parameters['linear_solver']= 'gmres'
	solver.parameters['preconditioner']= 'sor'
	solver.parameters['krylov_solver']['monitor_convergence']= True
	solver.parameters['krylov_solver']['absolute_tolerance']=1E-9
	solver.parameters['krylov_solver']['relative_tolerance']=1E-5
	solver.parameters['krylov_solver']['maximum_iterations']=MAX_ITERS
	solver.solve()
	print_roo('finished solving, calculating str??')

#Defining stress, strain, and strain energy in terms of the displacment u
	sigma_out = sigma(u)
	epsilon_out = eps(u)
	UU = 0.5*(sigma_out*epsilon_out)

#Depending on normal direction being observed, solution is projected into the function space and assembled over the mesh to output total strain energies in each direction
	if ii == 0:
		uxx = project(dot(dot(UU, nx), nx), FunctionSpace(mesh,'CG', 1))
		strXX = assemble(uxx*dx)
		print_roo(strXX)
	elif ii == 1:
		uxx = project(dot(dot(UU, ny), ny), FunctionSpace(mesh,'CG', 1))
		strYY = assemble(uxx*dx)
				#print_roo('strYY')
	elif ii == 2:
		uxx = project(dot(dot(UU, nz), nz), FunctionSpace(mesh,'CG', 1))
		strZZ = assemble(uxx*dx)
				#print_roo('strZZ')
	elif ii == 3:
		uxx = 0
		uhhx = project(dot(dot(UU, nx),nx), FunctionSpace(mesh, 'CG', 1))
		uhhy = project(dot(dot(UU, ny),ny), FunctionSpace(mesh, 'CG', 1))
		uhhz = project(dot(dot(UU, nz),nz), FunctionSpace(mesh, 'CG', 1))
		strHHx = assemble(uhhx*dx)
		strHHy = assemble(uhhy*dx)
		strHHz = assemble(uhhz*dx)
		strHH = strHHx + strHHy + strHHz
	else:
		print_roo('Error calculating str??')

#Cleaning up memory and variables to avoid overwrites
	del V
	del boundaries
	del bcx1
	del bcx2
	del bcy1
	del bcy2
	del bcz1
	del bcz2
	del bcx
	del u_
	del u
	del v
	del a
	del L
	del prblm
	del solver
	del sigma_out
	del epsilon_out
	del UU
	del uxx

print_roo('Done with all four calculations.')

#Use solution for post processing
#Mechanical properties extraction
U_Ux = strXX/(100**3)/1000000000
U_H = strHH/(100**3)/1000000000
vx = (3*U_Ux-U_H)/(-3*U_Ux-U_H)
nxx = 1-vx-2*(vx**2)
Ex = (2*nxx*U_Ux)/(1-vx)/((0.01**2))
U_Uy = strYY/(100**3)/1000000000
vy = (3*U_Uy-U_H)/(-3*U_Uy-U_H)
nyy = 1-vy-2*(vy**2)
Ey = (2*nyy*U_Uy)/(1-vy)/((0.01**2))
U_Uz = strZZ/(100**3)/1000000000
vz = (3*U_Uz-U_H)/(-3*U_Uz-U_H)
nzz = 1-vz-2*(vz**2)
Ez = (2*nzz*U_Uz)/(1-vz)/((0.01**2))
vAvg = (vx+vy+vz)/3  
EAvg = (Ex+Ey+Ez)/3  
Bulk = (2*U_H)/((0.03**2))

#Writing out material properties into output file
output = open(item, 'w')
output.write("%s %s %s %s %s %s %s %s %s %s\n" %("structure =", item, "volume=", volume, "E.avg=", EAvg, "v.avg=", vAvg, "Bulk=", Bulk))
output.close()

print_roo('Done with extracting mechanical properties!')

#Deleting leftover variables
del strZZ
del strYY
del strXX
del mesh
del surface
del U_Ux
del U_H
del U_Uy
del U_Uz
del vx
del nxx
del Ex
del vy
del nyy
del Ey
del vz
del nzz
del Ez
del vAvg
del EAvg
del Bulk
del strHH
del strHHx
del strHHy
del strHHz
del uhhx
del uhhy
del uhhz


print_roo('Finished!')
