using Distributions, NLsolve, Optim, QuantEcon, Plots

    # include a function implementing Rouwenhorst's algorithm to discretize an AR process
    include("rouwen.jl")

    

    # Input parameters of the model
    const A = 1 # productivity component in production function (we will not use it; only needed if one wants to slightly modify our function)
    const β = 0.96 # discount factor
    const ρ = 0.1 # death probability
    const α = 0.6 # curvature of production function 
    const θ = 0.5 # collaterazibility
    const δ_n = 0.2 # depreciation of new capital
    const δ_u = 0.2 # depreciation of old capital
    const γ = 1 
    const χ0 = 0.1 #Cost of raising equity parameters
    const χ1 = 5 # Cost of raising equity parameters
    const w0 = 5 # initial net worth
    const w_max = 30 # max value on grid for further plot
    const w_n = 50 # number of points on x-axis for further plot
    const ρ_s = 0.7 # persistence of schock in AR[1] process modeling idiosyncratic production shock
    const σ_u = 0.12 # standard deviation in AR[1] process 
    const σ_s = σ_u / sqrt(1 - ρ_s^2) # intermediate step to compute the mean of the AR[1] process
    const s_n = 2 # number of states for the shock
    const μ = -0.5 * σ_s^2 # mean of AR[1] process 
    const grid_temp, P_temp = rouwen(2,μ, σ_u, ρ_s) #use Rouwenhorst's method to return back possible states and probability transition matrix

    const s_grid = exp.(grid_temp) # build exponential grid for shock
    const Ps = P_temp
    const a = 0.5 # elasticity of output with respect to capital
    const ϵ = 5.0 # CES elasticity of substitution



    # Structure the parameters
    struct Parameters
        A::Float64
        β::Float64
        ρ::Float64
        α::Float64
        θ::Float64
        δ_n::Float64
        δ_u::Float64
        γ::Float64
        ρ_s::Float64
        σ_s::Float64
        s_grid::Vector{Float64}
        Ps::Matrix{Float64}
        w0::Float64
        a::Float64
        ϵ::Float64
        χ0::Float64
        χ1::Float64
    end

    # include functions that proceeds to policy function iterations
    include("Policy_Func_Iterations.jl")


    @doc raw"""
    This offers a precise documentation of the function run_model() by describing step by step what happens when someone runs the function.

    First, We build a structure with the following parameters given as inputs for the model.
        A: technology/productivity parameter on the production function 
         β: discount rate 
         ρ: exit probability
         α: curvature of the production function (power on capital stock) 
         θ: collaterazibility (if θ=0 there is no borrowing in the economy)
         δ_n: depreciation rate of new capital
         δ_u: depreciation rate of old capital
         γ: measure of first cohort of firms entering the economy
         ρ_s: persistence idiosyncratic productivity shock s ~ AR(1)
         σ _s: standard deviation innovation of the idiosyncratic productivity shock s
         s_grid: grid for the idiosyncratic productivity shock s 
         Ps: probability transition matrix (will follow form Rouwenhorst method)
         w0: initial net worth for new entrants
         a: elasticity of output with respect to capital
         ϵ: CES elasticity of substitution
         χ0: Cost of raising equity parameters
         χ1: Cost of raising equity parameters

    Then, we define a set of functions based on these parameters.
    - f(k) = Par.A * k^(Par.α) : this is a standard production function. It takes one input (capital stock k) and uses two of the parameters (the technology
    parameter A and the capital share of the production function α to compute the production output)
    - fk(k) = Par.α * Par.A * k^(Par.α - 1): this is the first derivative of the production function, with respect to the capital stock 
    which is the only input of the production function
    - fkk(k) = Par.α * Par.A * (Par.α - 1) * k^(Par.α - 2): second derivative of the production function with repect to the capital stock
    - fkinv(fk) = (fk / (Par.α * Par.A))^(1 / (Par.α - 1)) : inverse of the derivative function, useful for policy function iteration, the ouput is the
    capital stock 
    - g0(k_n, k_o) = Par.a^(1 / Par.α) * k_n .^((Par.ϵ - 1) / Par.ϵ) + (1 - Par.a)^(1 / Par.ϵ) * (Par.γ .* k_o) .^((Par.ϵ - 1) / Par.ϵ):
    computes the constant elasticity of substitution bundle for new and old capital for the production function with stocks of old and new capital as inputs
    - g(k_n, k_o) = g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1)) : this is the policy function which is an investment decision. Taking as inputs the old and new capital
    stocks, it gives as output the optimal amount of new capital to purchase at the next period
    - gn(k_n, k_o) = Par.a^(1 / Par.ϵ) * k_n .^((Par.ϵ - 1) / Par.ϵ - 1) * g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1) - 1) : 
    this is the derivative of the policy function with repect to the new capital, i.e. the marginal effect of investing in new capital on total capital in production
    - go(k_n, k_o) = Par.γ * (1 - Par.a)^(1 / Par.ϵ) * (Par.γ * k_o) .^((Par.ϵ - 1) / Par.ϵ - 1) * g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1) - 1):
    this is the derivative of the policy function with repect to the old capital, i.e. the marginal effect of investing in old capital on total capital in production

    We build a dictionary that contains both functions that we previously built and the parameters that enter these functions
    Fun = Dict(
            :g0 => g0,
            :f => f,
            :fk => fk,
            :fkk => fkk,
            :ϵ => Par.ϵ,
            :a => Par.a,
            :γ => Par.γ,
            :g => g,
            :gn => gn,
            :go => go
        )

        Then,  new parameters:
        ξ_grid1 : only useful for the competitive equilibrium
        ξpr_grid0 : only useful for the competitive equilibrium
        kU : matrix that gives values of old capital on the path to first best
        kN : matrix that gives values of new capital on the path to first best
        qstar : first-best valuation of old capital  
        damp : dampening parameter 
        diff_ξ : only useful for the competitive equilibrium
        tol_ξ : tolerance level when looking for the optimal price
        tol_q : tolerance level when looking for the optimal price
        maxiter_ξ : only useful for the competitive equilibrium
        maxiter_q : maximum number of iterations when computing the optimal price

        Again, we define a new set of parameters which will be our initial parameters, these are the one used originally by the authors of the paper as these parameters
        fasten convergence:
        q0 = 0.54
        kU0 = ones(50,2) .* [16 29]
        kN0 = ones(50,2) .* [16 29]
        kU = kU0
        kN = kN0

    Finally, we set these parameters at their initial value before running the loop around prices to find optimal prices
            xd : will be the distance
            iter_q : will defini the number of iterations already done inside the loop
            q1 : initial minimum price value
            q2 : initial maximum price value
            kN_fb : vector of new capital values. There are two colums to signify the levels with each realization of the idiosyncratic shock.
            kU_fb : vector of old capital values where (again) each column represent the realizations of the shock. 
            q_vec : initital price vector for old capital, will give the vector of prices for old capital obtained through the iteration loop
            q3 : init price value

    The loop aims to find the optimal price for old capital using policy function iteration. 
    The idea is to find the price such as demand and supply for the old capital are equal.
    To obtain such values, we use the getDS_shocks_fb function from the Policy_Func_Iterations file.

    First, we compute the difference between demand and supply of old capital for both the lowest possible price value (q1) and the highest one (q2).
    The differences in these situations will be called respectively xd1 and xd2.
    Then, we will want to know towards which direction price should move. Indeed, if the difference is different from 0, there is no clearing between and supply
    and either the prices are too high or too low. 
    This is why we then build a new price q3 as a function of q1 and q2 and the differences obtained in each situation. This is built such as the price that
    appears as clearing the most the market has a more important role in building q3.

    If the difference is the same in both cases (xd1=xd2), we just set q3 a bit below q1 as we cannot infer anything in such a situation.
    Otherwise, we apply a specific formula for q3 as mentioned before.
    Then, we compute demand and supply for this new value. If the difference xd3 is positive, it means demand and too high and prices were too low, then this is q1
    that will be changed by q3, i.e. we increase the lowest possible price. Conversly, if this is negative, demand is too low (or supply too high) and we simply diminish the highest possible price by
    replacing q2 by q3.
    However, if xd (which takes the value of xd3 at each period) reaches a low enough value, we consider that we have become close enough to the true price 
    of the first best and stop the loop.

    Finally, we compute some specific values such as the total capital at first best and the subsequent output in the economy.
    All the subsequent computations do not really matter in our situation, as we are in the first-best situation and that there are not different "states"
    at which values such as capital would differ.
"""
function run_model()
    

    w_grid = range(w0,w_max,w_n) # for further plots


    global Par = Parameters(A, β, ρ, α, θ, δ_n, δ_u, γ, ρ_s, σ_s, s_grid, Ps, w0, a, ϵ, χ0, χ1)

    # Create functions based on the parameters
    f(k) = Par.A * k^(Par.α) # production function
    fk(k) = Par.α * Par.A * k^(Par.α - 1) # derivative of production function
    fkk(k) = Par.α * Par.A * (Par.α - 1) * k^(Par.α - 2) # Second derivative of production function
    fkinv(fk) = (fk / (Par.α * Par.A))^(1 / (Par.α - 1)) # inverse of derivative (for policy function iteration)


    g0(k_n, k_o) = Par.a^(1 / Par.ϵ) * k_n .^((Par.ϵ - 1) / Par.ϵ) + (1 - Par.a)^(1 / Par.ϵ) * (Par.γ .* k_o) .^((Par.ϵ - 1) / Par.ϵ) # CES bundle for new and old capital for production function
    g(k_n, k_o) = g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1)) # policy function for capital 

    # marginal effect of investment in new and old capital on total capital in production
    gn(k_n, k_o) = Par.a^(1 / Par.ϵ) * k_n .^((Par.ϵ - 1) / Par.ϵ - 1) * g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1) - 1) 
    go(k_n, k_o) = Par.γ * (1 - Par.a)^(1 / Par.ϵ) * (Par.γ * k_o) .^((Par.ϵ - 1) / Par.ϵ - 1) * g0(k_n, k_o) .^(Par.ϵ / (Par.ϵ - 1) - 1)



    # Create dictionary for the functions
    global Fun = Dict(
        :g0 => g0,
        :f => f,
        :fk => fk,
        :fkk => fkk,
        :ϵ => Par.ϵ,
        :a => Par.a,
        :γ => Par.γ,
        :g => g,
        :gn => gn,
        :go => go
    )




    # Add some explanations here
    ξ_grid1 = zeros(w_n, 2)
    ξpr_grid0 = zeros(w_n, 2, 2) 
    kU = 3 * ones(w_n, 2) # vector for old capital
    kN = 5 * ones(w_n, 2) # vector for new capital
    qstar = 0.49 # first-best valuation of old capital
    damp = 0.5
    diff_ξ = 1.0
    tol_ξ = 1e-5
    tol_q = 1e-4
    maxiter_ξ = 100
    maxiter_q = 1000

    # init values at first best if ones wants faster convergence

    # vars = matread("src/cali_ce.mat")
    #q0 = vars[:"q_fb"]
    #kU0 = vars[:"kU_fb"]
    #kN0 = vars[:"kN_fb"]

    q0 = 0.54
    kU0 = ones(50,2) .* [16 29]
    kN0 = ones(50,2) .* [16 29]
    kU = kU0
    kN = kN0


    ### Compute First Best
    xd = 1
    iter_q = 0
    global q1 = 0.99 * qstar
    global q2 = 1.01 * qstar
    kN_fb = kN0[1, 1] * ones(1,2)
    kU_fb = kU0[1, 1] * ones(1,2)
    q_vec = [] # init price vector for old capital
    q3 = 0 # init price value

    # loop around prices to find optimal price

    global iter_q = 0

    while (iter_q < maxiter_q) && (maximum(abs.(xd)) > tol_q)
        
        global iter_q += 1

        # test with lowest possibile price value
        global q0 = q1

        # get optimal level of capital given current parameters values
        (kU_D_unc,kU_S_unc) = getDS_shocks_fb(Ps,s_grid, kN_fb,kU_fb,q0)

        # Compute the distance between the optimal values (we want it to be 0 at equilibrium as old capital is just former new capital)
        xd1 = kU_D_unc - kU_S_unc

        # test with highest possibile price value
        q0 = q2
        (kU_D_unc,kU_S_unc) = getDS_shocks_fb(Ps,s_grid, kN_fb,kU_fb,q0)
        xd2 = kU_D_unc - kU_S_unc

        # test whether the distance with highest and lowest price are the same (in which case, we try a bit below but not 
        # far from the lowest value)
        if xd2 == xd1
            global q3 = 0.9999q1
        else
            # otherwise: update price value by setting it to the intercept / slope ratio between  the prices (allows
            # to define a new price closer to the best of the former prices)
            slope = (xd2 - xd1) / (q2 - q1)
            intercept = xd2 - slope * q2
            global q3 = -intercept / slope
        end

        # try with this new value
        q0 = q3
        (kU_D_unc,kU_S_unc) = getDS_shocks_fb(Ps,s_grid, kN_fb,kU_fb,q0)
        xd3 = kU_D_unc - kU_S_unc
        xd = abs.(xd3)

        # just update lowest and upper bounds, depending on which one is binding us
        if xd3 > 0
            global q1 = q3
        else
            global q2 = q3
        end
        
        # Keep track of the price updates
        push!(q_vec, q3)
    end



    # init total capital at first Best
    k_fb  = ones(1,s_n)
    k_fb[:,1] = g(kN_fb[:,1], kU_fb[:,1])
    k_fb[:,2] = g(kN_fb[:,2], kU_fb[:,2])

    # output at first best
    Y_fb = (.5*(Ps[1,1]*s_grid[1] + Ps[1,2]*s_grid[2])*f(k_fb[1,1]) +   .5*(Ps[2,1]*s_grid[1] + Ps[2,2]*s_grid[2])*f(k_fb[1,2]))


    KN_fb = .5*kN_fb[1,1] +  .5*kN_fb[1,2]
    KU_fb = .5*kU_fb[1,1] +  .5*kU_fb[1,2]
    C_fb = Y_fb - δ_n*KN_fb
    Z_fb = Y_fb/((g(KN_fb, KU_fb))^α)

    # production from capital at first best in each state
    fk_1 = fk(k_fb[1])
    fk_2 = fk(k_fb[2])

    # mean marginal productivity of capital
    mpk_mean_fb = .5*(Ps[1,1]*log(s_grid[1]*fk_1) + Ps[1,2]*log(s_grid[2]*fk_1)) +   .5*(Ps[2,1]*log(s_grid[1]*fk_2) + Ps[2,2]*log(s_grid[2]*fk_2))

    # ... and standard deviation
    mpk_sd_fb  =( .5*Ps[1,1]*(log(s_grid[1]*fk_1) - mpk_mean_fb).^2 + .5*Ps[1,2]*(log(s_grid[2]*fk_1) - mpk_mean_fb).^2 + .5*Ps[2,1]*(log(s_grid[1]*fk_2) - mpk_mean_fb).^2 +.5*Ps[2,2]*(log(s_grid[2]*fk_2) - mpk_mean_fb).^2)^.5

    # output final values of capital at first best (want vector because need to make a choice from any point on the grid)
    # is the same everywhere though... because first-best
    kN_fb = ones(w_n,1).*kN_fb
    kU_fb = ones(w_n,1).*kU_fb
    k_fb = ones(w_n,1).*k_fb

    return (w_grid, kN_fb,kU_fb,k_fb)
end