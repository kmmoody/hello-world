#kangarooF
#author: Katherine Moody
#last edit: 7/10/2019
#FEniCS fea solver for mechanical properites of structures
#Beck Group
from __future__ import print_function
import os
import sys
import gc
from fenics import *
from mshr import *
from dolfin import *
from ufl import nabla_div
from mpi4py import MPI
item = sys.argv[1]
#import item
#import matplotlib.pyplot as plt
print('WELCOME TO KANGAROOF!!')
###Setting up user classes
#Creating boundaries for boundary conditions
#file_name= input(sys.argv[1])i
#output = open('mpi_test_1','w')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def print_roo(text):
    if rank == 0:
        print(text,flush=True)
    return
print_roo(item)
#if rank == 0:
#	file_name = input('Please type the name of your data output file: ')
#else:
#	file_name = None
#file_name = comm.bcast(file_name, root=0)


#if rank ==0:
#	mesh_list = input('Please type the file path to your structures-forward slashes!!: ')
#else:
#	mesh_list = None
#mesh_list = comm.bcast(mesh_list, root=0)
#mesh_list = input('Please type the file path where your structures are located-use forward slashes!!: ')
#def file_name():
#	if rank == 0:
#		input
#	mesh_list = input('Please type the file path of your structures - forward slashes!!!: ')
#else:
#	print_roo('error with input')
#	file_name = comm.bcast(file_name, root=0)
#	mesh_list = comm.bcast(mesh_list, root=0)
#else:
#	print_roo('error with input')
#output = open('B1aluminum','w')
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

###Variables and constants
#Creating variable to mark the facets of the boundaries
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

#material properties
E = 69e9 #elastic modulus
nu = 0.33 #poisson ratio
#lame properties
mu =Constant(E/2/(1+nu))
lmbda =Constant(E*nu/((1+nu)*(1-2*nu)))

#creating the normal directions for post processing
nx = Constant((1, 0, 0))
ny = Constant((0, 1, 0))
nz = Constant((0, 0, 1))

###User functions
#Strain
def eps(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
#Stress
def sigma(u):
    return lmbda*nabla_div(u)*Identity(3) + 2*mu*eps(u)
#strain energy
def en_dens(u):
    str_ele = eps(u)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC


###Loop preparations..
#Setting up list of structures...
#path_to_stls = "/scratch/kmmo264/C3/3/"
#file_list = sorted(os.listdir(path_to_stls))
#stl_list=[item for item in file_list if item.endswith('.off')]

###Loop!
#for item in stl_list:
	#try:
		#path_to_files = os.path.join(path_to_stls, item)
		#msg = "Starting structure {}..."
		#print_roo(msg.format(item))
surface = Surface3D(item)
mesh = generate_mesh(surface,0.5)
volume = assemble(1*dx(mesh))

	    #plot(mesh)
		#plt.show()
		#ax.auto_scale_xyz
for ii in range(4):
	gc.collect()
	msg = "Starting calculation {}..."
	print_roo(msg.format(ii)) 
###FIRST X
#Defining function space
	V = VectorFunctionSpace(mesh, 'Lagrange', 3)

#Creating variable for the searching of facets over the mesh
	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() -1)
#setting all the boundaries at value zero so they can be assigned a number
	boundaries.set_all(0)
#marking the facets of the boundaries and assigning them a number
	x500.mark(boundaries, 1)
	x600.mark(boundaries, 2)
	y500.mark(boundaries, 3)
	y600.mark(boundaries, 4)
	z500.mark(boundaries, 5)
	z600.mark(boundaries, 6)

#setting boundary conditions for uniaxial displacement x
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

#Creating functions for defining variables and solving
	print_roo('Setting up equations')
	v = TestFunction(V) 
	u_ = TrialFunction(V)

#creating our right hand and left hand side equations for solving
	a = inner(sigma(u_), eps(v))*dx
	L = dot(f, v)*dx + dot(T, v)*ds

	u = Function(V)
	print_roo('starting linear solve')
#solving for nodal displacement u
			#solve(a==L, u, bcx)
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

#defining stress, strain, and strain energy
	sigma_out = sigma(u)
	epsilon_out = eps(u)
	UU = 0.5*(sigma_out*epsilon_out)
			#print_roo('strain energy x calculation')
#projecting strain energy in the x direction and integrating over the mesh

	if ii == 0:
		uxx = project(dot(dot(UU, nx), nx), FunctionSpace(mesh,'CG', 1))
		strXX = assemble(uxx*dx)
		print_roo(strXX)
				#displacement = File(item + 'structuredisplacement.pvd')
				#displacement << u
				#energy = File(item + 'structurestrainenergy.pvd')
				#energy << uxx
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
			#plot(uxx)
			#plt.show()
			#print_roo(strXX)
			#print_roo('setting boundary conditions for y')
#deleting what I can	
			#mesh_file = File('fiberesh.pvd')
			#mesh_file << mesh
			#print_roo('deleting variables...')
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
			#del sigma_out
			#del epsilon_out

print_roo('Done with all four calculations.')
		#print_roo(strHH)
		#print_roo('beginning property calculations')
#calculations for bulk, stiffness, and poisson
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
		#print_roo("Ave. Poisson Ratio =", vAvg)
		#print_roo("Ave. Elastic Modulus =", EAvg)
		#print_roo("Bulk Modulus =", Bulk)
output = open(item, 'w')
output.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" %("structure =", item, "volume=", volume, "Ex=", Ex, "Ey=", Ey, "Ez=", Ez, "E.avg=", EAvg, "v.avg=", vAvg, "Bulk=", Bulk))
output.close()
		#output.write("%s" %item)
		#output.write("%s" %" volume=")
		#output.write("%s" %volume)
		#output.write("%s" %" E.avg=")
		#output.write("%s" %EAvg)
		#output.write("%s" %" v.avg=")
		#output.write("%s" %vAvg)
		#output.write("%s" %" Bulk=")
		#output.write("%s\n" %Bulk)

print_roo('Done with extracting mechanical properties!')
		#uzz = None
del strZZ
		#uyy = None
del strYY
		#uxx = None
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
	#except:
#surface = None
#mesh = None
#u = None
#volume = None
#boundaries = None
#u_ = None
#a = None
#v = None
#bx1 = None
#bx2 = None
#bz1 = None
#bz2 = None
#by1 = None
#		by2 = None
#		L = None
#		prblm = None
#		solver = None
#		print_roo('Caught Exception')
#		continue

print_roo('At end')