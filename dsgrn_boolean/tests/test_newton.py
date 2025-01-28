import unittest
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from dsgrn_boolean.utils.newton import newton_method, newton_method_autodiff

class TestNewtonMethod(unittest.TestCase):

    def setUp(self):
        # Example system of equations:
        # f1(x, y) = x^2 + y^2 - 4
        # f2(x, y) = x*y - 1
        def f(x):
            return np.array([x[0]**2 + x[1]**2 - 4,
                             x[0] * x[1] - 1])

        # Jacobian of f:
        def df(x):
            return np.array([[2*x[0], 2*x[1]],
                             [x[1], x[0]]])

        self.f = f
        self.df = df
        self.x0 = np.array([1.0, 2.0])
        self.tol = 1e-6

    def test_with_jacobian(self):
        x_sol, converged, iter_count = newton_method(self.f, self.x0, df=self.df, tol=self.tol)
        
        print("\nTest with Jacobian:")
        print(f"  Initial guess: {self.x0}")
        print(f"  Converged: {converged}")
        if converged:
            print(f"  Solution: {x_sol}")
            print(f"  Iterations: {iter_count}")
            print(f"  f(x_sol): {self.f(x_sol)}")  # Print the value of f at the solution
        
        self.assertTrue(converged)
        self.assertAlmostEqual(np.linalg.norm(self.f(x_sol)), 0.0, delta=self.tol)

    def test_without_jacobian(self):
        x_sol, converged, iter_count = newton_method(self.f, self.x0, tol=self.tol)

        print("\nTest without Jacobian (using finite differences):")
        print(f"  Initial guess: {self.x0}")
        print(f"  Converged: {converged}")
        if converged:
            print(f"  Solution: {x_sol}")
            print(f"  Iterations: {iter_count}")
            print(f"  f(x_sol): {self.f(x_sol)}")

        self.assertTrue(converged)
        self.assertAlmostEqual(np.linalg.norm(self.f(x_sol)), 0.0, delta=self.tol)

class TestNewtonMethodAutodiff(unittest.TestCase):
    def setUp(self):
        # Example system of equations (using JAX):
        def f(x):
            return jnp.array([x[0]**2 + x[1]**2 - 4,
                              x[0] * x[1] - 1])
        
        self.f = f
        self.x0 = jnp.array([1.0, 2.0])
        self.tol = 1e-6
    
    def test_autodiff(self):
        x_sol, converged, iter_count = newton_method_autodiff(self.f, self.x0, tol=self.tol)
        
        print("\nTest with automatic differentiation (using JAX):")
        print(f"  Initial guess: {self.x0}")
        print(f"  Converged: {converged}")
        if converged:
            print(f"  Solution: {x_sol}")
            print(f"  Iterations: {iter_count}")
            print(f"  f(x_sol): {self.f(x_sol)}")

        self.assertTrue(converged)
        self.assertAlmostEqual(jnp.linalg.norm(self.f(x_sol)), 0.0, delta=self.tol)

if __name__ == '__main__':
    unittest.main()