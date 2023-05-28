"""

"""
function fb_dyn_unc(xx::Array{Float64}, Par, Fun::Dict, q::Real, m::Real)
    
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


function getDS_shocks_fb(Ps::Matrix,s_grid, kN_fb,kU_fb,q0)

    # Performs fixed-point iteration to solve for the optimal level of capital.

    q = q0

    # update policy functions for all states of the shock
    for i_s = 1:s_n
       
        ξ = 0
        
        m_L = s_grid[1] # low-state value
        
        m_H = s_grid[2] # high-state value
        
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
