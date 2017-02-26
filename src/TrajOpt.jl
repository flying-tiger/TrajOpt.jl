module TrajOpt

    using FixedSizeArrays
    using Polynomials

    export HermiteBasis, QuadratureRule, BCTypes
    export compute_interpolation_matrices
    export assemble_stiffness_matrix
    export apply_bc!

    immutable Block{T,N} <: MutableFixedMatrix{T,N,N}

    end

    """
        hb = HermiteBasis(degree=N)

    Return left and right Hermite interpolation polynomials of specified degree.

    # Description
    Hermite basis function are used to interpolate a function over the interval
    τ = [0, 1] when the value of the function and it's first k derviatives are
    known at endpoints of the interval. The basis functions are simple τ-poly-
    nomials with degree N = 2k-1. To compute the value of the interpolant at a
    point within the interval, the basis functions are used to construct a
    weighted sum of the function data as follows (note: this is pseudo code):

        x_left  = [ x(0.0), x'(0.0) ... ]
        x_right = [ x(1.0), x'(1.0) ... ]
        x(τ) = sum( h.left(τ).*x_left ) + sum( h.right(τ).*x_right )

    # Fields
    * left::Vector{Poly}   Basis functions for left-endpoint function data
    * right::Vector{Poly}  Basis functions for right-endpoint function data
    """
    immutable HermiteBasis

        left  :: Vector{Poly{Rational{Int64}}}
        right :: Vector{Poly{Rational{Int64}}}

        function HermiteBasis(; degree=3)
            if degree == 1
                left  = [ Poly([ 1,  -1 ]//1, :τ) ]
                right = [ Poly([ 0,   1 ]//1, :τ) ]
            elseif degree == 3
                left  = [
                    Poly([ 1,   0,  -3,   2 ]//1, :τ)
                    Poly([ 0,   1,  -2,   1 ]//1, :τ)
                ]
                right = [
                    Poly([ 0,   0,   3,  -2 ]//1, :τ)
                    Poly([ 0,   0,  -1,   1 ]//1, :τ)
                ]
            elseif degree == 5
                left  = [
                    Poly([ 1,   0,   0, -10,  15,  -6 ]//1, :τ)
                    Poly([ 0,   1,   0,  -6,   8,  -3 ]//1, :τ)
                    Poly([ 0,   0,   1,  -3,   3,  -1 ]//2, :τ)
                ]
                right = [
                    Poly([ 0,   0,   0,  10, -15,   6 ]//1, :τ)
                    Poly([ 0,   0,   0,  -4,   7,  -3 ]//1, :τ)
                    Poly([ 0,   0,   0,   1,  -2,   1 ]//2, :τ)
                ]
            else
                throw(ArgumentError("degree=$degree: must be 1, 3, or 5."))
            end
            return new(left, right)
        end

    end


    """
        qr = QuadratureRule(num_points=N)

    Returns the N-point Guassian quadrature rule.

    # Description
    Numerical quadrature approximates the integral of a continuous function
    by sampling the integrand at discrete points in the domiain of integration
    and then taking a weighted sum of the sample values. Guassian quadrature
    rules are optimal for a 1D domain of integration: an N-point rule can
    exactly integrate a polynomial of degree 2*N-1. This type returns the
    sample locations and weights for a Gaussian Quadrature Rule with a
    specified number of sample locations.

    # Fields
    * locations::Vector{Real}  Quad point locations on bi-unit interval
    * weights::Vector{Real}    Summation weights for function samples

    """
    immutable QuadratureRule

        locations :: Vector{Float64}
        weights   :: Vector{Float64}

        function QuadratureRule(; num_points=2)
            if num_points == 1
                locations = [  0.0 ]
                weights   = [  2.0 ]
            elseif num_points == 2
                locations = [ -1.0,  1.0 ] * sqrt(1/3)
                weights   = [  1.0,  1.0 ]
            elseif num_points == 3
                locations = [ -1.0,  0.0,  1.0 ] * sqrt(3/5)
                weights   = [  5.0,  8.0,  5.0 ] / 9.0
            else
                throw(ArgumentError(
                    "num_points=$num_points: must be integer between 1 and 3."
                ))
            end
            return new(locations, weights)
        end
    end

    function compute_interpolation_matrices(hb, qr, time_points)

        # Preliminaries
        n_interval = length(time_points)-1
        n_quad = length(qr.locations)
        n_dof = length(hb.left)

        # Allocate interpolation matrices
        zero_block = MutableFixedMatrix{n_dof, n_quad, Float64}(0.0)
        HL = repmat([zero_block], n_interval)
        HR = repmat([zero_block], n_interval)

        # Map quadrature locations to unit interval
        τ = (qr.locations + 1)/2  # Maps [-1,1] => [0,1]
        τ_scaling = 1/2

        # Compute unit-interval interp matrices
        HL_unit = τ_scaling * polyval.( hb.left,  τ' )
        HR_unit = τ_scaling * polyval.( hb.right, τ' )

        # Apply interval scaling to interp matrices
        Δt = diff(time_points)
        scale_factors(Δt) = [ Δt^(q+1) for q = 0:n_dof-1 ]
        for i = 1:n_interval
            S = scale_factors(Δt[i]);
            HL[i] = S .* HL_unit
            HR[i] = S .* HR_unit
        end

        return (HL, HR)

    end

    """
        K = assemble_stiffness_matrix(hb, time)

    Compute stiffness matrix for time integration of 2nd order ODEs.

    # Description
    The Continuous Galerkin method can be used to discretize and solve systems
    of differential equations of the form ``\ddot{x} = f(t,x,\dot{x})``. The
    discretization of this system using Hermite basis functions results in a
    system of simultaneous non-linear equations ``Kx + f(x) = 0``, where ``K``
    is the stiffness matrix. This matrix is a block tri-diagonal matrix that
    is independent of the solution, x, and depends only on the distribution of
    time points used to discretize the solution domain. See the Jupyter note-
    book 'Theory.ipynb' for details.

    # Inputs
    * hb::HermiteBasis  Basis functions used to represent solution
    * time::AbstractVector{Float64}  Time points discretizing solution domain

    # Outputs
    * K::Tridiagonal{FixedSizeArrays.Mat}  Block tridiagonal stiffness matrix
    """
    function assemble_stiffness_matrix(hb, time_points)

        # Preliminaries
        n_point = length(time_points)
        n_interval = n_point-1
        n_dof = length(hb.left)

        # Error checking
        @assert n_dof>1   "Must use HermiteBasis with order >= 3"
        @assert n_point>2 "Must be at least 3 time points"

        # Allocate stiffness matrix
        zero_block = MutableFixedMatrix{Float64, n_dof, n_dof}(0.0)
        K = Tridiagonal(
            repmat( [zero_block], n_point-1 ), # Lower Diagonal
            repmat( [zero_block], n_point   ), # Main  Diagonal
            repmat( [zero_block], n_point-1 ), # Upper Diagonal
        )

        # Compute unit-interval stiffness matrices
        dhb_left  = polyder(hb.left)
        dhb_right = polyder(hb.right)
        integrate(x) = polyval( polyint(x), 1.0 )
        KLL = [ integrate(bi*bj) for bi=dhb_left,  bj=dhb_left  ]
        KLR = [ integrate(bi*bj) for bi=dhb_left,  bj=dhb_right ]
        KRL = [ integrate(bi*bj) for bi=dhb_right, bj=dhb_left  ]
        KRR = [ integrate(bi*bj) for bi=dhb_right, bj=dhb_right ]

        # Assemble global matrix from scaled unit-interval matrices
        Δt = diff(time_points)
        scale_factors(Δt) = [ Δt^-(p*q+1) for p=0:n_dof-1, q=0:n_dof-1 ]
        for i = 1:n_interval
            S = scale_factors(Δt[i])
            K[ i , i ] += S.*KLL
            K[ i ,i+1] += S.*KLR
            K[i+1, i ] += S.*KRL
            K[i+1,i+1] += S.*KRR
        end

        # Add boundary operators
        BL = [ polyval(bi*bj, 0.0) for bi=hb.left,  bj=dhb_left  ]
        BR = [ polyval(bi*bj, 1.0) for bi=hb.right, bj=dhb_right ]
        K[ 1 , 1 ] += scale_factors(Δt[ 1 ]) .* BL
        K[end,end] -= scale_factors(Δt[end]) .* BR

        return K

    end

    """
        F = assemble_load_vector(HL, HR, accel)

    Constructs the discrete load vector (RHS) for a uniform acceleration.

    # Description
    This function is a specicialized version of the assemble_load_vector that
    only works for constant accelerations that are independent of the trajectory
    solution (e.g. a constant gravitational acceleration). This is mainly used
    to verify correctness and accuracy of the Continuous Galerkin stiffness
    matrix and the associated boundary conditions.

    # Inputs
    HL::Vector{Mat}  Left-hand Hermite interpolation matrices
    HR::Vector{Mat}  Right-hand Hermite interpolation matrices
    accel::Float64   Constant acceleration applied to trajectory

    # Outputs
    F::Vector{Vec}   Load vector for discrete system of equations
    """
    function assemble_load_vector(HL, HR, weights, accel)

        # Preliminaries
        n_interval = length(HL)
        n_point = n_interval+1
        n_dof = size(HL[1],1)

        # Error checking
        @assert n_dof>1   "Must use HermiteBasis with order >= 3"
        @assert n_point>2 "Must be at least 3 time points"

        # Allocate load vector
        F = repmat(MutableFixedVector{Float64, n_dof}(0.0), n_point)

        # Evaluate load integrals
        aw = accel * weights
        for i = 1:n_interval
            F[ i ] += HL[i] * aw
            F[i+1] += HR[i] * aw
        end

    end

    """
    Defines named constants for different boundary condition types
    """
    module BCTypes
        @enum BCType dirichlet=1 neumann=2
    end


    """
        apply_bc!(K, bc, i)

    Modifies stiffness matrix to enforce Dirichlet/Neumman boundary conditions.

    # Description:
    Modifies stiffness matrix to enforce Dirichlet/Neumman boundary conditions
    at a specified time-point in the solution. Note that this doesn't actually
    define what the *value* at the boundary will be; the value is specified by
    calling apply_bc! with the load vector as the first argument, e.g.

        apply_bc!(f, bc, i, value)

    This version of apply_bc! simply modifies the stiffness matrix so that the
    required degree of freedom is set equal to the value specified in the load
    vector.
    """
    function apply_bc!{T <: MutableFixedMatrix}(K::Tridiagonal{T}, bc, i)
        # How this works:
        # To embed a boundary condition in the K matrix, we need to replace the
        # existing discrete equation for the "p"-th degree of freedom such that
        #   (K*x)[p] = x[p] = f[p]
        # Since K block tri-diagonal, this boils down to zero-ing out the
        # "q"-th row in of the three blocks coupled to the i-th node and then
        # setting the diagonal element to unity.
        #
        # Since we are using Hermite shape functions, the value of "q" depends
        # on the type of BC. If we want a Dirichlet BC we modify the 1st nodal
        # degree of freedom; if we want a Neumann BC, we modifiy the 2nd nodal
        # degree of freedom. This mapping from BC type to row index is codified
        # using the BCType enumeration.
        N = size(K,2)
        ibc = Int(bc)
        K[i,i][ibc,:]  = 0.0
        K[i,i][ibc,ibc] = 1.0
        if i > 1;  K[i,i-1][ibc,:] = 0.0;  end
        if i < N;  K[i,i+1][ibc,:] = 0.0;  end
        return nothing
    end


    """
        apply_bc!(F, bc, i, value)

    Modifies load vector to apply a Dirichlet/Neumman boundary conditions.

    # Description:
    Modifies assembled load vector to apply a Dirichlet or Neumman boundary
    conditions at a specified time-point in the solution. Note that for this
    to be effective, the stiffness matrix must also be modified at the specified
    node to accept a boundary condition. This is achieved by calling apply_bc!
    with the stiffness matrix as the first argument.
    """
    function apply_bc!{T <: MutableFixedVector}(F::Vector{T}, bc, i, value)
        F[i][Int(bc)] = value
        return nothing
    end


end
