module ReplicationLanteriRampini2023
#="""
Remark: This module uses the following packages: Distributions, NLsolve, Optim, QuantEcon, MAT, ForwardDiff, Plots. 
"""=#
using Distributions, NLsolve, Optim, QuantEcon, MAT, ForwardDiff, Plots

# include a function implementing Rouwenhorst's algorithm to discretize an AR process
include("rouwen.jl")

# include functions that proceeds to policy function iterations
include("Policy_Func_Iterations.jl")


"""
We build a structure with the parameters given as inputs for the model.
    - A: technology/productivity parameter on the production function 
    - β: discount rate 
    - ρ: exit probability
    - α: curvature of the production function (power on capital stock) 
    - θ: collaterazibility (if θ=0 there is no borrowing in the economy)
    - δ_n: depreciation rate of new capital
    - δ_u: depreciation rate of old capital
    - γ: measure of first cohort of firms entering the economy
    - ρ_s: persistence idiosyncratic productivity shock s ~ AR(1)
    - σ_s: standard deviation innovation of the idiosyncratic productivity shock s
    - s_grid: grid for the idiosyncratic productivity shock s 
    - Ps: probability transition matrix (will follow form Rouwenhorst method)
    - w0: initial net worth for new entrants
    - a: 
    - ϵ
    - χ0
    - χ1

"""
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

global w_grid = range(w0,w_max,w_n) # for further plots

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

Par = Parameters(A, β, ρ, α, θ, δ_n, δ_u, γ, ρ_s, σ_s, s_grid, Ps, w0, a, ϵ, χ0, χ1)


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

# init parameters with same values as authors (fastens the convergence process)
vars = matread("cali_ce.mat") ##need to "open folder" in visual studio and then choose "SANDBOX"

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
q0 = vars[:"q_fb"]
kU0 = vars[:"kU_fb"]
kN0 = vars[:"kN_fb"]
kU = kU0
kN = kN0


### Compute First Best
xd = 1
iter_q = 0
q1 = 0.99 * qstar
q2 = 1.01 * qstar
kN_fb = kN0[1, 1] * ones(1,2)
kU_fb = kU0[1, 1] * ones(1,2)
q_vec = [] # init price vector for old capital
q3 = 0 # init price value



# loop around prices to find optimal price
iter_q =0

while (iter_q < maxiter_q) && (maximum(abs.(xd)) > tol_q)
    
    iter_q += 1

    # test with lowest possibile price value
    q0 = q1

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
        q3 = 0.9999q1
    else
        # otherwise: update price value by setting it to the intercept / slope ratio between  the prices (allows
        # to define a new price closer to the best of the former prices)
        slope = (xd2 - xd1) / (q2 - q1)
        intercept = xd2 - slope * q2
        q3 = -intercept / slope
    end

    # try with this new value
    q0 = q3
    (kU_D_unc,kU_S_unc) = getDS_shocks_fb(Ps,s_grid, kN_fb,kU_fb,q0)
    xd3 = kU_D_unc - kU_S_unc
    xd = abs.(xd3)

    # just update lowest and upper bounds, depending on which one is binding us
    if xd3 > 0
        q1 = q3
    else
        q2 = q3
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

# Plot for capital both old and new
l = @layout [a;b ;  c]


p1 = plot(w_grid, kN_fb[:,1], linewidth=2)
plot!(p1,w_grid, kN_fb[:,2], linewidth=2)
xlabel!(p1,"w")
ylabel!(p1,"kN")


p2 = plot(w_grid, kU_fb[:,1], linewidth=2)
plot!(p2,w_grid, kU_fb[:,2], linewidth=2)
xlabel!(p2,"w")
ylabel!(p2,"kU")


p3 = plot(w_grid, k_fb[:,1], linewidth=2)
plot!(p3,w_grid, k_fb[:,2], linewidth=2)
xlabel!(p3,"w")
ylabel!(p3,"k")


plot(p1,p2,p3,layout = l)

end
