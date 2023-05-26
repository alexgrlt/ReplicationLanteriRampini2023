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

function getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)

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

                global xx = nlsolve(objective!, xx0)

                global flag_con = xx.f_converged

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

        global w_grid = collect(w_grid)

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

    global m
    global ξpr
    global ξ
    global w

    return (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) 

end




# -------- test ---- --------- #

function getDS_shocks(Ps,Fun,s_grid, maxiter_ξ, tol_ξ,q0,ξ_grid1,ξpr_grid0)

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
    global wpr
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

                global x = value(x)
                global y = value(y)

                global xx = [x,y]

                global flag_con = has_values(model)

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

                    global x = value(x)
                    global y = value(y)

                    global xx = [x,y]

                    global flag_con = has_values(model)
                    kNpr, kUpr = xx[1], xx[2]
                    flag_con_vec[j_w, i_s] = flag_con
                    val_con_vec[j_w, i_s] = maximum(abs.(val_con))
                    _, τ_n_con, τ_u_con, ϕ_agent_con, _,bb = ce_dyn_con_τ([x,y],Par, Fun, q, m,ξ, ξpr, w, η, τ)
                    append!(xx, bb)

                    global xx
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
                global wpr
            end
        end

        global w_grid = collect(w_grid)

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

        global wpr
    end

    λ = get_λ_convex(w_grid,wpr)

    kU_D = λ*δ_u*kU[:]

    kU_S = λ*δ_n*kN[:]

    kU_D = kU_D[1] # use the [1] so that not 1*1 vector anymore but float

    kU_S = kU_S[1] # use the [1] so that not 1*1 vector anymore but float

    kU_D_unc = δ_u*(kU[w_n,1] + kU[w_n,2])/2
    kU_S_unc = δ_n*(kU[w_n,2] + kN[w_n,2])/2


    #### For debug Alex

    global m
    global ξpr
    global ξ
    global w

    return (kU_D_unc, kU_S_unc,λ,kU,kN,kU_D,kU_S,wpr,flag_con,xx) 

end



