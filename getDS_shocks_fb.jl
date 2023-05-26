function getDS_shocks_fb(Ps::Matrix,s_grid, kN_fb,kU_fb,q0)

    q = q0

    for i_s = 1:s_n
    
        xi = 0
        
        m_L = s_grid[1]
        
        m_H = s_grid[2]
        
        m = dot(Ps[i_s,:], [m_L, m_H])
        
        
        xx0 = [kN_fb[i_s], kU_fb[i_s]]
        
        
        xx = nlsolve(xx -> fb_dyn_unc(xx, Par, Fun, q, m), xx0, opts=Optim.Options(show_trace=false))
        
        kNpr = xx.zero[1]
        kUpr = xx.zero[2]
        
        kN_fb[i_s] = kNpr
        kU_fb[i_s] = kUpr
    
    end
    
    kN_fb[2] = kNpr
    kU_fb[2] = kUpr
    
    kU_D_unc = delta_u * (kU_fb[1] + kU_fb[2]) / 2
    
    kU_S_unc = delta_n * (kN_fb[1] + kN_fb[2]) / 2

    return (kU_D_unc,kU_S_unc)
end