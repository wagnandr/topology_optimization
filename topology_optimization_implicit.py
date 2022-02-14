import math
import time
import random
import dolfin as df


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
    
    def setup_solvers(self):
        pass

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


if __name__ == '__main__':
    N = 32 
    num_time_steps = 2**10 

    tau_value = 1e-8 
    tau = df.Constant(tau_value)

    eps_value = 1. / 16 / math.pi 
    eps = df.Constant(eps_value)

    gamma_F = df.CompiledSubDomain('near(x[1], 0.0) && (0.75 <= x[0] && x[0] <= 1.)')
    gamma_D = df.CompiledSubDomain('x[0] - 1e-10 < -1 && x[0] + 1e-10 > -1')

    f = df.Constant((0, -250))

    mu1 = df.Constant(5000)
    lambda1 = df.Constant(5000)

    mesh = df.RectangleMesh(df.Point(-1, 0), df.Point(1, 1), N, N)

    solver = CahnHilliardProblem(mesh, tau, eps, gamma_F, f, gamma_D, mu1, lambda1)

    solve_problem = False 

    checkpoint_file = df.XDMFFile('output/checkpoint_phi.xdmf')

    if solve_problem:
        phi_init = df.Function(solver.function_space.sub(1).collapse())
        phi_init.interpolate(InitialConditions())
        df.assign(solver.u_now.sub(1), phi_init)
        solver.setup_solvers()

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
        phi = df.Function(V)
        checkpoint_file.read_checkpoint(phi, 'phi', 0)

        elasticity_solver = LinearElasticityProblem(mesh, gamma_D, gamma_F, f, mu1, lambda1, eps) 

        u = elasticity_solver.solve(phi)

        energy_form = elasticity_solver.get_energy(u, phi)
        energy = df.project(energy_form, V)

        solution_file_phi = df.File('output/phi_noise.pvd')
        solution_file_u = df.File('output/u_noise.pvd')
        solution_file_energy = df.File('output/energy_noise.pvd')

        solution_file_phi.write(phi, 0)
        solution_file_u.write(u, 0)
        solution_file_energy.write(energy, 0)

