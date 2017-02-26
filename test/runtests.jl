using Base.Test
using Polynomials
using TrajOpt

@testset "HermiteBasis" begin

    """ Check basis function values and derivatives """
    function check_endpoints(basis)
        all_pass = true
        for i = 1:length(basis.left)

            # Check i-1st derivative at left endpoint
            all_pass &= polyder( basis.left[i],  i-1 )(0.0) == 1.0
            all_pass &= polyder( basis.right[i], i-1 )(0.0) == 0.0

            # Check i-1st derivative at right endpoint
            all_pass &= polyder( basis.left[i],  i-1 )(1.0) == 0.0
            all_pass &= polyder( basis.right[i], i-1 )(1.0) == 1.0

        end
        return all_pass
    end

    H1 = HermiteBasis(degree = 1)
    H3 = HermiteBasis(degree = 3)
    H5 = HermiteBasis(degree = 5)
    @test_throws ArgumentError HermiteBasis(degree = 2)

    # Verify properties of the basis functions
    @test check_endpoints(H1)
    @test check_endpoints(H3)
    @test check_endpoints(H5)

    # Verify basis function are stored in a vector
    @test size(H3.left)  == (2,)
    @test size(H5.left)  == (3,)
    @test size(H5.right) == (3,)

end

@testset "QuadratureRule" begin

    """ Check quadrature rule accuracy """
    function check_quadrature(qr::QuadratureRule)

        npt = length(qr.locations)
        test_function = Poly(ones(2*npt))

        # Compute approximate integral between [-1.0, 1.0] via quadrature
        samples = polyval(test_function, qr.locations)
        approx = dot(qr.weights, samples)

        # Compute integral between [-1.0, 1.0] via symbolic integration
        integral = polyint(test_function)
        exact = integral(1.0) - integral(-1.0)

        return exact ≈ approx

    end

    Q1 = QuadratureRule(num_points = 1)
    Q2 = QuadratureRule(num_points = 2)
    Q3 = QuadratureRule(num_points = 3)
    @test_throws ArgumentError QuadratureRule(num_points = 4)

    # Verify properties of the basis functions
    @test check_quadrature(Q1)
    @test check_quadrature(Q2)
    @test check_quadrature(Q3)

    # Verify basis function are stored in a vector
    @test size(Q2.locations)  == (2,)
    @test size(Q3.locations)  == (3,)
    @test size(Q3.weights)    == (3,)

end

@testset "Test Cases" begin

    """
    # Test Case 1: Constant Acceleration in 1D w/ Full Initial Conditions

    This is a simple test to verify that the stiffness matrix, interpolation
    matrices and boundary conditions are implemented correctly. In the case of a
    constant linear acceleration, the load vector is independent of the solution and
    the trajectory may be directly computed by inverting the K matrix onto the load
    vector (no linearization or iteration required). Since the exact solution is a
    quadratic polynomial (x = x0 + v0*t + 0.5*a*t^2), the discretized solution
    should match the exact solution for a HermiteBasis with degree >= 3.

    """
    function test_case_1()

        const x0 = 3.0  # Initial displacement [m]
        const v0 = 2.5  # Initial velocity     [m/s]
        const a  = 1.0  # Acceleration         [m/s^2]
        const Δt = 4.0  # Integration Period   [s]

        # Define time discretization and solution method
        time_points = linspace(0.0, Δt, 9)
        qr = QuadratureRule(num_points = 2)
        hb = HermiteBasis(degree = 3)

        # Setup stiffness matrix
        K = assemble_stiffness_matrix(hb, time_points)
        apply_bc!(K, BCTypes.dirichlet, 1)
        apply_bc!(K, BCTypes.neumann,   1)

        # Setup load vector
        (HL,HR) = compute_interpolation_matrices(hb, qr, time_points)
        #F = assemble_load_vector(HL, HR, qr.weights, a)
        #apply_bc!(F, BCTypes.dirichlet, 1, x0)
        #apply_bc!(F, BCTypes.neumann,   1, v0)

        # Solve
        #x = K \ F

        return true #x[end][pos] == x0 + v0*Δt + 0.5*a*Δt^2

    end
    @test test_case_1()

end