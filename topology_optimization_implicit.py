import math
import time
import random
import dolfin as df
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


class InitialConditions(df.UserExpression):
    """ Random initial conditions """
    def __init__(self, **kwargs):
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.0 + 0.02*(0.5 - random.random())


def mobility(phi):
    phi = df.conditional(df.gt(phi, 1), 1, phi)
    phi = df.conditional(df.lt(phi, -1), -1, phi)
    return 9/4 * (1 - phi*phi) * (1 - phi*phi)


def epsilon(u):
    return df.Constant(0.5)*(df.grad(u) + df.grad(u).T)


class FullEquation(df.NonlinearProblem):
    def __init__(self, a, L, bcs):
        df.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def F(self, b, x):
        df.assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b)

    def J(self, A, x):
        df.assemble(self.a, tensor=A, keep_diagonal=True)
        for bc in self.bcs:
            bc.apply(A)


def cut(x):
    x = df.conditional(df.gt(x, 1), 1, x)
    x = df.conditional(df.lt(x, -1), -1, x)
    return x


class CahnHilliardProblem:
    def __init__(self, mesh, tau, eps, gamma_F, f, gamma_D, mu1, lambda1):
        P1ElVec = df.VectorElement('P', mesh.ufl_cell(), 1)
        P1El = df.FiniteElement('P', mesh.ufl_cell(), 1)
        El = df.MixedElement([P1ElVec, P1El, P1El])

        self.function_space = df.FunctionSpace(mesh, El)

        self.u_next = df.Function(self.function_space, name='u_next')
        self.u_now = df.Function(self.function_space, name='u_now')

        boundary_marker = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        gamma_F.mark(boundary_marker, 1)

        self.ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_marker)

        self.f = f

        self.tau = tau
        self.eps = eps
        self.gamma = df.Constant(1)

        self.mu1 = mu1
        self.lambda1 = lambda1

        u_D = df.Constant((0, 0))
        self.bcs = [df.DirichletBC(self.function_space.sub(0), u_D, gamma_D)]
    
    def forms(self):
        _, phi_now, _ = df.split(self.u_now)

        u, phi_next, mu = df.split(self.u_next)
        
        v, psi, nu = df.TestFunctions(self.function_space)

        tau, eps = self.tau, self.eps

        mu_diff = df.Constant((1 - self.eps**2)) * self.mu1
        lambda_diff = df.Constant((1 - self.eps**2)) * self.lambda1

        energy = - 2 * mu_diff * df.inner(epsilon(u), epsilon(u)) + lambda_diff * df.div(u) * df.div(u)


        mu_le = df.Constant(0.5) * ((cut(phi_next) + df.Constant(1)) * self.mu1 - (cut(phi_next) - df.Constant(1)) * df.Constant(self.eps**2) * self.mu1)
        lambda_le = df.Constant(0.5) * ((cut(phi_next) + df.Constant(1)) * self.lambda1 - (cut(phi_next) - df.Constant(1)) * df.Constant(self.eps**2) * self.lambda1)

        a_form = 0

        # add linear elasticity 
        a_form += 2 * mu_le * df.inner(epsilon(u), epsilon(v)) * df.dx + lambda_le * df.div(u) * df.div(v) * df.dx
        a_form += - df.inner(self.f, v) * self.ds(1)

        # add cahn-hilliard:
        a_form += df.inner(eps/tau * (phi_next - phi_now), psi) * df.dx
        a_form += mobility(phi_now) * df.inner(df.grad(mu), df.grad(psi)) * df.dx

        a_form += mu * nu * df.dx
        a_form += - self.gamma * eps * df.inner(df.grad(phi_next), df.grad(nu)) * df.dx
        a_form += - self.gamma/eps * (2 * phi_next - 3 * phi_now + phi_now**3 ) * nu * df.dx
        a_form += - energy * nu * df.dx

        du = df.TrialFunction(self.function_space)
        a_diff_form = df.derivative(a_form, self.u_next, du)

        return a_diff_form, a_form 

    def solve(self):
        a, L = self.forms()
        pb = FullEquation(a, L, self.bcs)

        df.assemble(a)

        solver= df.PETScSNESSolver('newtonls')
        df.PETScOptions.set('ksp_type', 'gmres') 
        df.PETScOptions.set('pc_type', 'bjacobi')
        df.PETScOptions.set('sub_pc_type', 'ilu')
        df.PETScOptions.set('sub_pc_factor_levels', 10)
        df.PETScOptions.set('pc_factor_mat_solver_type', 'mumps')
        df.PETScOptions.set('snes_atol', 1e-8)
        solver.parameters['report'] = False 
        solver.set_from_options()

        self.u_next.vector()[:] = self.u_now.vector()[:]
        solver.solve(pb, self.u_next.vector())
        self.u_now.vector()[:] = self.u_next.vector()[:]


class LinearElasticityProblem:
    def __init__(self, mesh, gamma_D, gamma_F, f, mu1, lambda1, eps):
        self.mesh = mesh
        self.function_space = df.VectorFunctionSpace(mesh, 'P', 1)

        boundary_marker = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        gamma_F.mark(boundary_marker, 1)

        self.ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_marker)

        self.mu1 = mu1 
        self.lambda1 = lambda1 

        self.eps = eps

        self.f = f

        u_D = df.Constant((0, 0))
        self.bcs = [df.DirichletBC(self.function_space, u_D, gamma_D)]
    
    def forms(self, phi):
        u = df.TrialFunction(self.function_space)
        v = df.TestFunction(self.function_space)
        a = self.get_bilinear_form(u, v, phi) 
        l = self.get_linear_form(v) 
        return a, l
    
    def get_bilinear_form(self, u, v, phi):
        mu_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.mu1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.mu1)
        lambda_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.lambda1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.lambda1)
        return 2 * mu_le * df.inner(epsilon(u), epsilon(v)) * df.dx + lambda_le * df.div(u) * df.div(v) * df.dx

    def get_linear_form(self, v):
        return df.inner(self.f, v) * self.ds(1)

    def get_energy(self, u, phi):
        mu_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.mu1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.mu1)
        lambda_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.lambda1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.lambda1)
        return 2 * mu_le * df.inner(epsilon(u), epsilon(u)) + lambda_le * df.div(u) * df.div(u)
    
    def solve(self, phi, u_sol=None):
        a, l = self.forms(phi)
        if u_sol is None:
            u_sol = df.Function(self.function_space, name='u')
        df.solve(a == l, u_sol, self.bcs, solver_parameters={
            #'linear_solver': 'cg', 
            #'preconditioner': 'hypre_amg'
        } ) 
        return u_sol


def generate_perturbation_(V, dim_grid_y):
    dim_grid_y -= 1
    v2d = df.vertex_to_dof_map(V)
    #g = df.Function(V)
    x, y = np.meshgrid(np.linspace(-4, +4, dim_grid_y * 8), np.linspace(-2, +4, dim_grid_y*4), indexing='ij')
    #sigma, l1, l2 = 0.1, 0.05, 0.1
    sigma, l1, l2 = 0.1, 0.1 * 10, 0.1 * 10
    g_reshape = sigma * np.exp(- x*x/l1**2 - y*y/l2**2 )
    #g.interpolate(df.Expression('sigma * exp( - (x[0]*x[0]) / (l1*l1) - (x[1]*x[1])/ (l2*l2) )', sigma=0.1, l1=0.05, l2=0.1, degree=1))
    #g.interpolate(df.Expression('sigma * exp( - (x[0]*x[0]) / (l1*l1) - (x[1]*x[1])/ (l2*l2) )', sigma=0.1, l1=0.05, l2=0.1, degree=1))
    #g_reorder = g.vector()[v2d]
    #g_reshape = g_reorder.reshape((dim_grid_y, -1))
    noise = np.random.normal(0, 1, g_reshape.shape)
    g_hat = np.fft.rfftn(g_reshape, s=g_reshape.shape, axes=(0,1))
    noise_hat = np.fft.rfftn(noise, s=g_reshape.shape, axes=(0,1))
    eta = np.fft.irfftn(g_hat * noise_hat, s=(dim_grid_y+1, 2*dim_grid_y+1), axes=(0,1))
    eta_fun = df.Function(V)
    eta_fun.vector()[v2d] = eta.flatten()[:]
    df.plot(eta_fun)
    plt.show()

    return eta_fun


def generate_perturbation(V, dim_grid_y, sigma=1., l1=1, l2=1):
    dim_grid_y -= 1
    v2d = df.vertex_to_dof_map(V)
    x, y = np.meshgrid(np.linspace(-4, +4, dim_grid_y * 8), np.linspace(-2, +4, dim_grid_y*4), indexing='ij')
    #g_reshape = sigma * np.exp(- x*x/l1**2 - y*y/l2**2 )
    g_reshape = sigma * np.exp(- np.sqrt(x*x/l1**2 + y*y/l2**2) )
    noise = np.random.normal(0, 1, g_reshape.shape)
    g_hat = np.fft.rfftn(g_reshape, s=g_reshape.shape, axes=(0,1))
    noise_hat = np.fft.rfftn(noise, s=g_reshape.shape, axes=(0,1))
    eta = np.fft.irfftn(g_hat * noise_hat, s=(dim_grid_y+1, dim_grid_y+1), axes=(0,1))
    eta_fun = df.Function(V)
    eta_fun.vector()[v2d] = eta.flatten()[:]
    return eta_fun


if __name__ == '__main__':
    N = 32 
    num_time_steps = 2**12 

    tau_value = 1e-8 
    tau = df.Constant(tau_value)

    eps_value = 1. / 16 / math.pi 
    eps = df.Constant(eps_value)

    gamma_F = df.CompiledSubDomain('near(x[1], 0.0) && (0.75 <= x[0] && x[0] <= 1.)')
    gamma_D = df.CompiledSubDomain('x[0] - 1e-10 < -1 && x[0] + 1e-10 > -1')

    f = df.Constant((0, -250))

    mu1 = df.Constant(5000)
    lambda1 = df.Constant(5000)

    #mesh = df.RectangleMesh(df.Point(-1, 0), df.Point(1, 1), 2*N, N)
    mesh = df.RectangleMesh(df.Point(-1, 0), df.Point(1, 1), N, N)

    solver = CahnHilliardProblem(mesh, tau, eps, gamma_F, f, gamma_D, mu1, lambda1)

    solve_problem = False 

    checkpoint_file = df.XDMFFile('output/checkpoint_phi.xdmf')

    if solve_problem:
        phi_init = df.Function(solver.function_space.sub(1).collapse())
        phi_init.interpolate(InitialConditions())
        df.assign(solver.u_now.sub(1), phi_init)

        solution_file_u = df.File('output/u.pvd')
        solution_file_phi = df.File('output/phi.pvd')
        solution_file_u.write(solver.u_now.split(True)[0], 0)
        solution_file_phi.write(solver.u_now.split(True)[1], 0)

        start = time.time()
        t = 0
        for it_t in range(1,num_time_steps+1):
            t += tau_value

            print('it = {}/{}, t = {}'.format(it_t, num_time_steps, t))

            solver.solve()

            tau_value *= 1.01 
            tau.assign(tau_value)

            solution_file_u.write(solver.u_now.split(True)[0], t)
            solution_file_phi.write(solver.u_now.split(True)[1], t)
        
        checkpoint_file.write_checkpoint(solver.u_now.split(True)[1], 'phi', 0, df.XDMFFile.Encoding.HDF5, False)

        print('needed {}s'.format((time.time()-start)))
    else:
        V = solver.function_space.sub(1).collapse()
        phi_orig = df.Function(V, name='phi')
        checkpoint_file.read_checkpoint(phi_orig, 'phi', 0)

        phi_orig.vector()[:] = np.clip(phi_orig.vector()[:], -1, +1)

        elasticity_solver = LinearElasticityProblem(mesh, gamma_D, gamma_F, f, mu1, lambda1, eps) 

        solution_file_phi = df.File('output/phi_noise.pvd')
        solution_file_u = df.File('output/u_noise.pvd')
        solution_file_energy = df.File('output/energy_noise.pvd')

        phi_vertex_values =  phi_orig.compute_vertex_values()
        mesh.coordinates()
        #mymap = df.vertex_to_dof_map(V)
        #mymap_inv = np.argsort(mymap)
        #print('> ', np.linalg.norm(phi_vertex_values[:] - phi_orig.vector()[mymap]) )
        #print('> ', np.linalg.norm(phi_vertex_values[mymap_inv] - phi_orig.vector()[:]) )
        # print(phi_vertex_values)
        # print(mesh.coordinates())

        list_phi_samples = []

        # get the original energy:
        u_orig = elasticity_solver.solve(phi_orig)
        u_orig.rename('u', '')

        energy_form_orig = elasticity_solver.get_energy(u_orig, phi_orig)
        energy_orig = df.project(energy_form_orig, V)
        energy_orig.rename('energy', '')

        energy_value_orig = df.assemble(energy_form_orig * df.dx)
        print('energy = {}'.format(energy_value_orig))

        solution_file_u_orig = df.File('output/u_orig.pvd')
        solution_file_phi_orig = df.File('output/phi_orig.pvd')
        solution_file_energy_orig = df.File('output/energy_orig.pvd')
        solution_file_phi_orig.write(phi_orig, 0)
        solution_file_u_orig.write(u_orig, 0)
        solution_file_energy_orig.write(energy_orig, 0)

        # evaluate the samples:
        for i in range(10):
            phi = phi_orig.copy(True)
            phi.rename('phi', '') 

            a = phi.vector()[:]
            a = np.arctanh(a)
            a += generate_perturbation(V, N+1, sigma=1e-3, l1=0.1, l2=0.1).vector()[:]
            a = np.tanh(a)
            phi.vector()[:] = a

            u = elasticity_solver.solve(phi)
            u.rename('u', '')

            energy_form = elasticity_solver.get_energy(u, phi)
            energy = df.project(energy_form, V)
            energy.rename('energy', '')

            energy_value = df.assemble(energy_form * df.dx)
            print('energy = {}'.format(energy_value))

            solution_file_phi.write(phi, i)
            solution_file_u.write(u, i)
            solution_file_energy.write(energy, i)

            list_phi_samples.append(phi)
        
        phi_mean = df.Function(V, name='phi_mean')
        phi_mean_array = np.mean(np.array([phi.vector()[:].tolist() for phi in list_phi_samples]), axis=0)
        phi_mean.vector()[:] = phi_mean_array 

        phi_mean_file = df.File('output/phi_mean.pvd')
        phi_mean_file.write(phi_mean, 0)
