"""
fb_dyn_unc(xx::Array{Float64}, Par::Parameters, Fun::Dict, q::Real, m::Any)
Returns an array zz with the first-order conditions for new and old capital at the optimal allocation (see equations 38 and 39 in the paper) that will be used in the policy function iteration.
The expression of these FOC is given explicitly as in 
The inputs correspond to an array with some level of new and old capital (xx), the parameter structure defined before with the associated dictionary with functions at the core of the setup (notably the policy function with its derivatives and production function with its derivatives), a price for capital q and another parameter m. 
In the next function (getDS_shocks_fb), m is defined as the matrix of expected payoffs.

"""
function fb_dyn_unc(xx::Array{Float64}, Par::Parameters, Fun::Dict, q::Real, m::Any)
    
    # Gives the first-order conditions for the optimal allocation of capital.
    # zz[1] is the FOC associated to equation 38 in the paper.
    # zz[2] is the FOC associated to equation 39 in the paper.

    zz = ones(2,1)
    k_n = xx[1]
    k_o = xx[2]
    ξ = 0 
    ξp = 0 

    k = Fun[:g](k_n, k_o)
    z=1
    
    zz[1] = -(1+ξ) + Par.β*(z*Fun[:fk](k)*Fun[:gn](k_n,k_o)*m + (1-Par.δ_n)*(1+(1-Par.ρ)*ξp)) + Par.β*Par.δ_n*q*(1+(1-Par.ρ)*ξp)
    zz[2] = -q*(1+ξ) + Par.β*(z*Fun[:go](k_n,k_o)*Fun[:fk](k)*m + (1-Par.δ_u)*q*(1+(1-Par.ρ)*ξp))

    return zz
end

"""
getDS_shocks_fb(Ps::Matrix,s_grid, kN_fb,kU_fb,q0) 
This function uses NLsolve to solve for the FOC found with fb_dyn_unc ans deduce the level of old capital supplied and demanded optimally by firms of a given cohort. 
Its arguments are: Ps, the transition matrix of the idiosyncratic shock found usinf Rouwenhorst method; s_grid, theexponential grid of the shock; kN_fb and kU_fb the levels of new and old capital, and q0 an initial price of capital. 
It returns a tuple with the demand and the supply of old capital. 

"""
function getDS_shocks_fb(Ps::Matrix,s_grid::Matrix, kN_fb::Array,kU_fb::Array,q0::Number)

    q = q0

    # update policy functions for all states of the shock
    for i_s = 1:s_n  ## all states of idiosyncratic shock 
       
        ξ = 0
        
        m_L = s_grid[1] # low-state value
        
        m_H = s_grid[2] # high-state value, Rouwenhorst is made with N=2 only
        
        m = Ps[i_s,:]' * [m_L, m_H] # matrix of expected payoffs
        
        # give current values of capital 
        xx0 = [kN_fb[i_s], kU_fb[i_s]]
        
        # Solves the FOC, using the initial level of capital as the starting point in the algorithm.
        xx = nlsolve(xx -> fb_dyn_unc(xx, Par, Fun, q, m), xx0)
        
        # get back to results of the optimization program
        kNpr = xx.zero[1]
        kUpr = xx.zero[2]
        
        # update first-best values of capital and 
        kN_fb[i_s] = kNpr
        kU_fb[i_s] = kUpr
    
    end
    
    # Output the depreciated stocks of capital (but why /2)
    kU_D_unc = δ_u * (kU_fb[1] + kU_fb[2]) / 2
    
    kU_S_unc = δ_n * (kN_fb[1] + kN_fb[2]) / 2

    return (kU_D_unc,kU_S_unc)
end
