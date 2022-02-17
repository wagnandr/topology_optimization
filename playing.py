import scipy 
import scipy.optimize
import numpy as np
import dolfin as df
from matplotlib import pyplot as plt
import math

from topology_optimization_implicit import (
    LinearElasticityProblem,
    epsilon,
    cut,
    InitialConditions
)


def potential(phi):
    return 1/8 * (1 - phi**2)**2
    # return 1/8 * abs(1 - phi**2)
    #return 1/8 * df.exp( (1 - phi**2)**2 )


df.set_log_level(df.LogLevel.ERROR)


class EvaluateOptimizationQuantities:
    def __init__(self, mesh, eps, gamma_F, f, gamma_D, mu1, lambda1, penalty, gamma):
        El = df.FiniteElement('P', mesh.ufl_cell(), 1)
        self.function_space = df.FunctionSpace(mesh, El)

        self.eps = eps
        self.gamma = gamma 

        self.mu1 = mu1
        self.lambda1 = lambda1

        self.penalty = penalty 

        self.phi = df.Function(self.function_space)
        self.phi0 = df.Function(self.function_space)
        #self.phi0.interpolate(df.Constant('0'))
        self.phi0.interpolate(InitialConditions())

        self.elasticity_solver = LinearElasticityProblem(mesh, gamma_D, gamma_F, f, mu1, lambda1, eps)

    def assemble_energy(self, phi_vec):
        phi = self.phi
        phi.vector()[:] = phi_vec
        u = self.elasticity_solver.solve(self.phi)

        mu_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.mu1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.mu1)
        lambda_le = df.Constant(0.5) * ((cut(phi) + df.Constant(1)) * self.lambda1 - (cut(phi) - df.Constant(1)) * df.Constant(self.eps**2) * self.lambda1)

        form = 0
        form += self.gamma * eps * df.inner(df.grad(phi), df.grad(phi)) * df.dx
        form += self.gamma/eps * potential(phi) * df.dx
        form += 2 * mu_le * df.inner(epsilon(u), epsilon(u)) * df.dx + lambda_le * df.div(u) * df.div(u) * df.dx

        mass = df.assemble(phi * df.dx(domain=self.function_space.mesh()))

        energy = df.assemble(form)
        energy += penalty * (0 - mass ) ** 2

        print(f'energy {energy}, mass {mass}, min {phi_vec.min()}, max {phi_vec.max()}')

        return energy

    
    def assemble_gradient_l2(self, phi_vec):
        phi = self.phi
        phi.vector()[:] = phi_vec
        u = self.elasticity_solver.solve(self.phi)
        psi = df.TrialFunction(self.function_space)

        mu_diff = df.Constant((1 - self.eps**2)) * self.mu1
        lambda_diff = df.Constant((1 - self.eps**2)) * self.lambda1

        form = 0
        form += self.gamma * eps * df.inner(df.grad(phi), df.grad(psi)) * df.dx
        form += self.gamma/eps * df.derivative(potential(phi), phi, psi) * df.dx
        form += - df.Constant(0.5) * (2 * mu_diff * df.inner(epsilon(u), epsilon(u)) + lambda_diff * df.div(u) * df.div(u)) * psi * df.dx

        mass = df.assemble(phi * df.dx(domain=self.function_space.mesh()))

        form += penalty * 2 * (0 - mass) * (-psi) * df.dx(domain=self.function_space.mesh())

        vec = df.assemble( form )
        return np.array(vec[:])
    
    def create_function(self, name=None, numpy_vector=None):
        u = df.Function(self.function_space)
        if name is not None:
            u.rename(name, '')
        if numpy_vector is not None:
            u.vector()[:] = numpy_vector[:]
        return u
    
    def create_energy_functional(self):
        return lambda phi_vec, *args: self.assemble_energy(phi_vec)
        
    def create_gradient_l2_functional(self):
        return lambda phi_vec, *args: self.assemble_gradient_l2(phi_vec)


class Save:
    def __init__(self, filename, optimizer):
        self.it = 0
        self.file = df.File(filename)
        self.optimizer = optimizer

    def __call__(self, vector):
        print(f'it = {self.it}')
        self.file.write(self.optimizer.create_function('phi', vector), self.it)
        self.it += 1


if __name__ == '__main__':
    N = 32

    eps_value = 1. / 16 / math.pi 
    eps = df.Constant(eps_value)

    gamma = df.Constant(1e+0)

    #gamma_F = df.CompiledSubDomain('near(x[0], 1.)')
    gamma_F = df.CompiledSubDomain('near(x[1], 0.0) && (0.75 <= x[0] && x[0] <= 1.)')
    gamma_D = df.CompiledSubDomain('x[0] - 1e-10 < -1 && x[0] + 1e-10 > -1')

    f = df.Constant((0, -250))
    #f = df.Constant((+250, 0))

    mu1 = df.Constant(5000)
    lambda1 = df.Constant(5000)

    penalty = 5e1

    mesh = df.RectangleMesh(df.Point(-1, 0), df.Point(1, 1), N, N)

    grad = EvaluateOptimizationQuantities(mesh, eps, gamma_F, f, gamma_D, mu1, lambda1, penalty, gamma=gamma)

    num_dof = len(grad.phi0.vector()[:])

    res = scipy.optimize.minimize(
        fun=grad.create_energy_functional(),
        jac = grad.create_gradient_l2_functional(),
        #method = 'BFGS',
        #method = 'Nelder-Mead',
        #method = 'CG',
        method = 'L-BFGS-B',
        x0=np.array(grad.phi0.vector()[:]),
        options={
            'disp': True, 
            'maxiter': 1000,
            'gtol': 1e-16,
            'ftol': 1e-16,
            'maxcor': 1,
            'maxls': 100
        },
        bounds=scipy.optimize.Bounds(-np.ones(num_dof), +np.ones(num_dof), True),
        callback=Save(filename='output/phi_scipy.pvd', optimizer=grad)
    )
    
    print(res.success)
    print(res.message)
    solution = grad.create_function()
    solution.vector()[:] = res.x
