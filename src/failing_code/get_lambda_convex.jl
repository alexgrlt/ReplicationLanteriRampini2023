function get_λ_convex(w_grid,wpr)

    # Compute stationary distribution of markov process

    i_wpr = zeros(w_n,2) #w_n is the nb of points on x-axis
    P_state_temp = zeros(w_n, w_n, 2, 2)
    w_grid_step = w_grid[2] - w_grid[1]

    # init matrices
    i_wpr_down = Array{Int64}(undef, w_n, 2, 2) 
    i_wpr_up = Array{Int64}(undef, w_n, 2, 2) 
    pr_trans_down = zeros(w_n, 2, 2)
    pr_trans_up = zeros(w_n, 2, 2)

    ω_up = 0
    ω_down = 0


    # create transition probability matrix

    for i_w in 1:w_n 
        for i_s in 1:2
            for i_sp in 1:2
                if wpr[i_w, i_s, i_sp] <= w_grid[end]
                    wpr_vec = wpr[i_w, i_s, i_sp] .* ones(w_n, 1)
                    wpr_diff = wpr_vec - w_grid # got rid of the transpose
                    
                    wpr_diff = wpr_diff[:] # Alex idea: to convert to array

                    w_grid_down_temp = findall(wpr_diff .>= 0.)
                    i_wpr_temp_down = w_grid_down_temp[end]

                    w_grid_up_temp = findall(wpr_diff .<= 0.)
                    i_wpr_temp_up = w_grid_up_temp[1]

                    ω_up = (wpr[i_w, i_s, i_sp] - w_grid[i_wpr_temp_down]) / w_grid_step
                    ω_down = 1 - ω_up

                    i_wpr_down[i_w, i_s, i_sp] = i_wpr_temp_down
                    i_wpr_up[i_w, i_s, i_sp] = i_wpr_temp_up

                else
                    i_wpr_down[i_w, i_s, i_sp] = w_n
                    i_wpr_up[i_w, i_s, i_sp] = w_n
                end

                pr_trans_down[i_w, i_s, i_sp] = ω_down * Ps[i_s, i_sp]

                pr_trans_up[i_w, i_s, i_sp] = ω_up * Ps[i_s, i_sp]

                P_state_temp[i_w, i_wpr_down[i_w, i_s, i_sp], i_s, i_sp] = pr_trans_down[i_w, i_s, i_sp]
                
                P_state_temp[i_w, i_wpr_up[i_w, i_s, i_sp], i_s, i_sp] = pr_trans_up[i_w, i_s, i_sp]
            end
        end
    end

        P_state = [P_state_temp[:,:,1,1] P_state_temp[:,:,1,2]; P_state_temp[:,:,2,1] P_state_temp[:,:,2,2]] #this is the transition matrix for the state space (i think)

        λ00 = zeros(1, w_n*2)
        λ00[1] = 0.5
        λ00[w_n+1] = 0.5
        λ0 = λ00


        # get invariant distribution
        diff_λ = 1 
        tol_λ = 10^-6
        maxiter_λ = 500
        iter_λ = 0 

    while diff_λ>tol_λ && iter_λ<maxiter_λ
        iter_λ += 1
        λ1 = λ0*P_state
        diff_λ = maximum(abs.(λ1*(1-ρ) + ρ*λ00 - λ0))
        λ0 = λ1*(1-ρ) + ρ*λ00

    end
    
    λ = λ0/sum(λ0) # invariant distribution over the state space

    return λ
end 
