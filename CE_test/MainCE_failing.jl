#=
This code aims to replicate the competitive equilibrium results of the paper. 
It relies on  the function getDS_shocks that does not work with NLsolve as it does not converge.
    We tried to use JuMP and Ipopt in getDS_shocks1 but it is way too long (15h+) and lead to wrong answer. 
=#


using Distributions, NLsolve, Optim, QuantEcon, MAT, ForwardDiff, Interpolations, Plots,JuMP,Ipopt,Calculus

# include a function implementing Rouwenhorst's algorithm to discretize an AR process
include("rouwen.jl")

# include a function that gives the cost of dividends to firms
include("Cost_function.jl")

# and inverse cost function
include("cpinv.jl")

# include the function that gets lambda (parameter of the markov chain)

include("get_lambda_convex.jl")


# include FOC for capital 
include("ce_dyn_unc_τ.jl")

# include FOC for capital
include("ce_dyn_con_τ.jl")



function getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)
    
    w_grid = range(w0,w_max,w_n)
    iter_ξ = 0
    diff_ξ = 1

    q = q0

    flag_con_vec = zeros(w_n, s_n)
    flag_unc_vec = zeros(w_n, s_n)
    val_con_vec = zeros(w_n, s_n)
    val_unc_vec = zeros(w_n, s_n)
    bmin_mat = zeros(w_n, s_n)
    ξpr_grid1 = zeros(w_n,2, 2)
    τN = zeros(w_n, s_n)
    τU = zeros(w_n, s_n)
    b = zeros(w_n, s_n)
    ϕ = zeros(w_n, s_n)
    wpr = vars["wpr"]
    ξpr_mat = zeros(w_n,s_n)

    while iter_ξ < maxiter_ξ && diff_ξ > tol_ξ
        
        iter_ξ += 1
        ξ_grid0 = ξ_grid1
    
        for i_s = 1:s_n
            spr = Ps[i_s,:] * s_grid'
            for i_w = 1:w_n
                j_w = w_n + 1 - i_w
                w = w_grid[j_w]
                ξpr = Ps[i_s,:]' * reshape(ξpr_grid0[j_w, i_s, :], (2,1))
                ξpr = Float64(ξpr[1])
                ξpr_mat[j_w, i_s] = ξpr
                ξ = (1-ρ) * ξpr
                m_L = s_grid[1] * (1+ξpr_grid0[j_w, i_s, 1]*(1-ρ))
                m_H = s_grid[2] * (1+ξpr_grid0[j_w, i_s, 2]*(1-ρ))
                m = Ps[i_s,:]' * [m_L m_H]'
                
                m = m[1] #unsure about this (I also added the transpose above)
                
                if iter_ξ < 2
                    xx0 = min.([kN[j_w, i_s] kU[j_w, i_s]], [w/3 w/3])
                else
                    xx0 = [kN[j_w, i_s] kU[j_w, i_s]]
                end
                
                ps = Ps[i_s,:]
                # [xx0 d[j_w, i_s]] [zz, τ_n, τ_u, ϕ_agent, bp] = ce_dyn_unc_τ(xx,Par, Fun, q, m, xip, w, η, τ)
                
                function objective!(xx) # create a closure for the function of interest
                    zz, _, _, _, _ = ce_dyn_unc_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    xx = zz #  because NLsolve algorithm needs to update value of the function 
                end
                
                #xx, val_con, flag_con = fsolve(objective!, xx0) from matlab

                
                 xx = nlsolve(objective!, xx0)

                flag_con = xx.f_converged

                xx = xx.zero

                xx = [xx[1],xx[2]]

                function val_opt(xx) 
                    zz, _, _, _, _ = ce_dyn_unc_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    return zz
                end

                val_con = val_opt(xx)
                append!(xx, Cpinv(ξ, Par))

                kNpr, kUpr = xx[1], xx[2]
                flag_unc_vec[j_w, i_s] = flag_con
                val_unc_vec[j_w, i_s] = maximum(abs.(val_con))
                _, τ_n_temp, τ_u_temp, _, bp_temp = ce_dyn_unc_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    
                bmin = -β*θ*(kNpr*(1-δ_n*(1-q)) + q*kUpr*(1-δ_u))
                
                function objectivebis!(xx) # create a closure for the function of interest
                    ce_dyn_con_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    xx = zz # because NLsolve algorithm needs to update value of the function
                end
                
                if bp_temp < bmin
                    xx, val_con, flag_con = nlsolve(objectivebis!, xx0)
                    kNpr, kUpr = xx[1], xx[2]
                    flag_con_vec[j_w, i_s] = flag_con
                    val_con_vec[j_w, i_s] = maximum(abs.(val_con))
                    _, τ_n_temp, τ_u_temp, _, bp_temp = ce_dyn_con_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                end
                
                
                kNpr = xx[1]
                kUpr = xx[2]
                _, ξ, _ = Cost(xx[3], Par)
                bpr = bp_temp

                kN[j_w, i_s] = kNpr
                kU[j_w, i_s] = kUpr
                bmin_mat[j_w, i_s] = -β * θ * (kNpr * (1 - δ_n * (1 - q)) + q * kUpr * (1 - δ_u))

                k[j_w, i_s] = Fun[:g](kNpr, kUpr)
                b[j_w, i_s] = bpr
                d[j_w, i_s] = w - kNpr - q * kUpr - bpr
                ξ_grid1[j_w, i_s] = ξ[1] # ξ is a 1*1 matrix

                

                ϕ[j_w, i_s] = max(ξ[1] - (1 - ρ) * ξpr, 0)

                


                τN[j_w, i_s] = τ_n_temp
                τU[j_w, i_s] = τ_u_temp

                # bpr = min(bmin, bpr)

                wpr[j_w, i_s, 1] = kNpr * (1 - δ_n * (1 - q)) + kUpr * (1 - δ_u) * q + bpr / β + s_grid[1] * f(k[j_w, i_s])
                wpr[j_w, i_s, 2] = kNpr * (1 - δ_n * (1 - q)) + kUpr * (1 - δ_u) * q + bpr / β + s_grid[2] * f(k[j_w, i_s])
            end
        end

          w_grid = collect(w_grid)

        ξpr_grid1[:,1,1] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1)));extrapolation_bc=0).(wpr[:,1,1])
        ξpr_grid1[:,1,2] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1)));extrapolation_bc=0).(wpr[:,1,2])
        ξpr_grid1[:,2,1] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1)));extrapolation_bc=0).(wpr[:,2,1])
        ξpr_grid1[:,2,2] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1)));extrapolation_bc=0).(wpr[:,2,2])
        
        #ξpr_grid1[:,1,1] = interp1(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1))), wpr[:,1,1];extrapvalue=nothing)
        #ξpr_grid1[:,1,2] = interp1(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1))), wpr[:,1,2];extrapvalue=nothing)

        #ξpr_grid1[:,2,1] = interp1(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1))), wpr[:,2,1];extrapvalue=nothing)
        #ξpr_grid1[:,2,2] = interp1(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1))), wpr[:,2,2];extrapvalue=nothing) #previously extrapvalue was 0

        diff_ξ = maximum(abs.(ξpr_grid0[:] .- ξpr_grid1[:]))
        diff_ξ_today = maximum(abs.(ξ_grid0[:] .- ξ_grid1[:]))

        ξpr_grid0 = damp.*ξpr_grid1 .+ (1 .- damp).*ξpr_grid0
    end

    λ = get_λ_convex(w_grid,wpr)

    kU_D = λ*δ_u*kU[:]

    kU_S = λ*δ_n*kN[:]

    kU_D = kU_D[1] # use the [1] so that not 1*1 vector anymore but float

    kU_S = kU_S[1] # use the [1] so that not 1*1 vector anymore but float

    kU_D_unc = δ_u*(kU[w_n,1] + kU[w_n,2])/2
    kU_S_unc = δ_n*(kU[w_n,2] + kN[w_n,2])/2

    
    #### For debug Alex

   #= global m
    global ξpr
    global ξ
    global w
=#
    return (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) 

end

# -------- test JuMP---- --------- #

function getDS_shocks1(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)

    iter_ξ = 0
    diff_ξ = 1

    q = q0

    flag_con_vec = zeros(w_n, s_n)
    flag_unc_vec = zeros(w_n, s_n)
    val_con_vec = zeros(w_n, s_n)
    val_unc_vec = zeros(w_n, s_n)
    bmin_mat = zeros(w_n, s_n)
    ξpr_grid1 = zeros(w_n,2, 2)
    τN = zeros(w_n, s_n)
    τU = zeros(w_n, s_n)
    b = zeros(w_n, s_n)
    ϕ = zeros(w_n, s_n)
    wpr = vars["wpr"]
    #global wpr
    ξpr_mat = zeros(w_n,s_n)

    while iter_ξ < maxiter_ξ && diff_ξ > tol_ξ
        
        iter_ξ += 1
        ξ_grid0 = ξ_grid1
    
        for i_s = 1:s_n
            spr = Ps[i_s,:] * s_grid'
            for i_w = 1:w_n
                j_w = w_n + 1 - i_w
                w = w_grid[j_w]
                ξpr = Ps[i_s,:]' * reshape(ξpr_grid0[j_w, i_s, :], (2,1))
                ξpr = Float64(ξpr[1])
                ξpr_mat[j_w, i_s] = ξpr
                ξ = (1-ρ) * ξpr
                m_L = s_grid[1] * (1+ξpr_grid0[j_w, i_s, 1]*(1-ρ))
                m_H = s_grid[2] * (1+ξpr_grid0[j_w, i_s, 2]*(1-ρ))
                m = Ps[i_s,:]' * [m_L m_H]'
                
                m = m[1] #unsure about this (I also added the transpose above)
                
                if iter_ξ < 2
                    xx0 = min.([kN[j_w, i_s] kU[j_w, i_s]], [w/3 w/3])
                else
                    xx0 = [kN[j_w, i_s] kU[j_w, i_s]]
                end
                
                ps = Ps[i_s,:]
                # [xx0 d[j_w, i_s]] [zz, τ_n, τ_u, ϕ_agent, bp] = ce_dyn_unc_τ(xx,Par, Fun, q, m, xip, w, η, τ)
                
                # Compute norm of residuals of the objective
                function obj(a,b)
                    xx =[a,b]
                    zz, _, _, _, _ = ce_dyn_unc_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    if sum(abs.(zz)) == NaN
                        zz= [100.,100.]
                    elseif sum(abs.(zz)) == -Inf
                        zz= [100.,100.]
                    end
                    return sum(abs.(zz))
                end
                
                model = Model(Ipopt.Optimizer)
                set_silent(model)
                set_optimizer_attribute(model, "max_iter", 100)

                @variable(model,x >= 0.5)
                @variable(model,y >= 0.5)
                register(model,:obj,2,obj;autodiff = true)
                @NLobjective(model, Min,obj(x,y))

                set_start_value(x, 25.0)
                set_start_value(y, 25.0)

                optimize!(model)

                solution_summary(model;verbose=true)

                x = value(x)
                y = value(y)

                 xx = [x,y]

                 flag_con = has_values(model)

                function val_opt(xx) 
                    zz, _, _, _, _ = ce_dyn_unc_τ(xx,Par, Fun, q, m, ξpr, w, η, τ)
                    return zz
                end

                val_con = val_opt(xx)
                append!(xx, Cpinv(ξ, Par))

                kNpr, kUpr = x, y
                flag_unc_vec[j_w, i_s] = flag_con
                val_unc_vec[j_w, i_s] = maximum(abs.(val_con))
                _, τ_n_temp, τ_u_temp, _, bp_temp = ce_dyn_unc_τ([x,y],Par, Fun, q, m, ξpr, w, η, τ)
                    
                bmin = -β*θ*(kNpr*(1-δ_n*(1-q)) + q*kUpr*(1-δ_u))

                if bp_temp < bmin


                    function objbis(a,b)

                        xx =[a,b]
                        
                        zz, _, _, _, _,_ = ce_dyn_con_τ(xx,Par, Fun, q, m,ξ, ξpr, w, η, τ)
                        
                        zz = [zz[1],zz[2]]
                        
                        if sum(abs.(zz)) == NaN
                            zz= [100.,100.]
                        elseif sum(abs.(zz)) == -Inf
                            zz= [100.,100.]
                        end
                        return sum(abs.(zz))
                    end
                    
                    model = Model(Ipopt.Optimizer)
                    set_silent(model)
                    set_optimizer_attribute(model, "max_iter", 100)

                    @variable(model,x >= 0.5)
                    @variable(model,y >= 0.5)

                    register(model,:objbis,2,objbis;autodiff = true)
                    @NLobjective(model, Min,objbis(x,y))

                    set_start_value(x, 25.0)
                    set_start_value(y, 25.0)

                    optimize!(model)

                    solution_summary(model;verbose=true)

                    x = value(x)
                    y = value(y)

                    xx = [x,y]

                    flag_con = has_values(model)
                    kNpr, kUpr = xx[1], xx[2]
                    flag_con_vec[j_w, i_s] = flag_con
                    val_con_vec[j_w, i_s] = maximum(abs.(val_con))
                    _, τ_n_con, τ_u_con, ϕ_agent_con, _,bb = ce_dyn_con_τ([x,y],Par, Fun, q, m,ξ, ξpr, w, η, τ)
                    append!(xx, bb)

                    xx
                end
                
                
                kNpr = x
                kUpr = y
                _, ξ, _ = Cost(xx[3], Par)
                bpr = bp_temp

                kN[j_w, i_s] = kNpr
                kU[j_w, i_s] = kUpr
                bmin_mat[j_w, i_s] = -β * θ * (kNpr * (1 - δ_n * (1 - q)) + q * kUpr * (1 - δ_u))

                k[j_w, i_s] = Fun[:g](kNpr, kUpr)
                b[j_w, i_s] = bpr
                d[j_w, i_s] = w - kNpr - q * kUpr - bpr
                ξ_grid1[j_w, i_s] = ξ[1] # ξ is a 1*1 matrix

                

                ϕ[j_w, i_s] = max(ξ[1] - (1 - ρ) * ξpr, 0)

                


                τN[j_w, i_s] = τ_n_temp
                τU[j_w, i_s] = τ_u_temp

                # bpr = min(bmin, bpr)

                wpr[j_w, i_s, 1] = kNpr * (1 - δ_n * (1 - q)) + kUpr * (1 - δ_u) * q + bpr / β + s_grid[1] * f(k[j_w, i_s])
                wpr[j_w, i_s, 2] = kNpr * (1 - δ_n * (1 - q)) + kUpr * (1 - δ_u) * q + bpr / β + s_grid[2] * f(k[j_w, i_s])
                wpr
            end
        end

        w_grid = collect(w_grid)

        ξpr_grid1[:,1,1] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1)));extrapolation_bc=0).(wpr[:,1,1])
        ξpr_grid1[:,1,2] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1)));extrapolation_bc=0).(wpr[:,1,2])
        ξpr_grid1[:,2,1] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1)));extrapolation_bc=0).(wpr[:,2,1])
        ξpr_grid1[:,2,2] = LinearInterpolation(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1)));extrapolation_bc=0).(wpr[:,2,2])
        
        #ξpr_grid1[:,1,1] = interp1(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1))), wpr[:,1,1];extrapvalue=nothing)
        #ξpr_grid1[:,1,2] = interp1(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1))), wpr[:,1,2];extrapvalue=nothing)

        #ξpr_grid1[:,2,1] = interp1(w_grid, vec(reshape(ξ_grid1[:,1], (w_n,1))), wpr[:,2,1];extrapvalue=nothing)
        #ξpr_grid1[:,2,2] = interp1(w_grid, vec(reshape(ξ_grid1[:,2], (w_n,1))), wpr[:,2,2];extrapvalue=nothing) #previously extrapvalue was 0

        diff_ξ = maximum(abs.(ξpr_grid0[:] .- ξpr_grid1[:]))
        diff_ξ_today = maximum(abs.(ξ_grid0[:] .- ξ_grid1[:]))

        ξpr_grid0 = damp.*ξpr_grid1 .+ (1 .- damp).*ξpr_grid0

        wpr
    end

    λ = get_λ_convex(w_grid,wpr)

    kU_D = λ*δ_u*kU[:]

    kU_S = λ*δ_n*kN[:]

    kU_D = kU_D[1] # use the [1] so that not 1*1 vector anymore but float

    kU_S = kU_S[1] # use the [1] so that not 1*1 vector anymore but float

    kU_D_unc = δ_u*(kU[w_n,1] + kU[w_n,2])/2
    kU_S_unc = δ_n*(kU[w_n,2] + kN[w_n,2])/2


    #= ###For debug Alex

    global m
    global ξpr
    global ξ
    global w
    =#

    return (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) 

end


## ------- Actual computations ----- #

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

w_grid = range(w0,w_max,w_n) # for further plots

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
vars = matread("CE_test/cali_ce.mat") ##need to "open folder" in visual studio and then choose "SANDBOX"

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


## competitive equilibrium
τ = 0. # init tax level (no tax for competitive equilibrium)
η = 0. 
k = ones(w_n, 2) # init capital
d = zeros(w_n,2) # init dividend streams

xd =  1 # init difference

iter_q = 0 # init iteration flag

maxiter_q = 200

q_fb = vars[:"q_fb"]
q1 = q_fb # init at first-best value

q2 = q_fb*1.02

q10 = q_fb*1.02
q20 = q_fb
xd10 = 1
xd20 = 1
tol_ξ = 10^-6


while (iter_q < maxiter_q) && (maximum(abs.(xd)) > tol_q)
    
    iter_q += 1
    q0 = q1
    
    if q0 == q10
        
        xd1 = xd10
    else   
        (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) = getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)
        
        xd1 = kU_D - kU_S
        
    end 
    
    q10=  q1
    xd10 = xd1
    
    q0 = q2
    
    if q2 == q20
        
        xd2 =  xd20
    else
        
        (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) = getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)
        
        xd2 = kU_D - kU_S
    end
    
    q20 = q2
    
    xd20 = xd2
    
    slope = (xd2 - xd1)/(q2-q1)
    intercept = xd2 - slope * q2
    q3 = -intercept/slope
    
    q0 = q3
    
    (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr) = getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)
    
    xd3 = kU_D - kU_S
    
    xd = abs(xd3)
    
    if xd3>0
        q1 = q3
    else
        q2 = q3
    end
    
    q3
    #w_grid = range(w0,w_max,w_n)
    w_grid
    wpr
end


q_eq = q3


plot(w_grid, λ[1:w_n])
plot!(w_grid, λ[w_n+1:end])



# Plot for capital both old and new
l = @layout [a;b ;  c;d]

kU = vars[:"kU_eq"]
kN = vars[:"kN_eq"]
k  = kU + kN


p1 = plot(w_grid, kN[:,1], linewidth=2)
plot!(p1,w_grid, kN[:,2], linewidth=2)
xlabel!(p1,"w")
ylabel!(p1,"kN")


p2 = plot(w_grid, kU[:,1], linewidth=2)
plot!(p2,w_grid, kU[:,2], linewidth=2)
xlabel!(p2,"w")
ylabel!(p2,"kU")


p3 = plot(w_grid, k[:,1], linewidth=2)
plot!(p3,w_grid, k[:,2], linewidth=2)
xlabel!(p3,"w")
ylabel!(p3,"k")




p4 = plot(w_grid, wpr[:,1,1], linewidth=2)
plot!(p4,w_grid, wpr[:,2,2], linewidth=2)
plot!(p4,w_grid, w_grid, linestyle=:dash)
xlabel!(p4,"w")
ylabel!(p4,"wpr")


plot(p1,p2,p3,p4,layout = l)