function ce_dyn_unc_τ(xx, Par, Fun, q, m, ξp, w, η, τ)

    τ = τ[1]

    zz = xx
    
    k_n = xx[1]
    k_o = xx[2]
    
    ξ = (1-Par.ρ)*ξp[1]
    
    d = Cpinv(ξ, Par)
    _, ξ, _ = Cost(d, Par)
    ξ = ξ[1]
    
    z = 1
    
    k = Fun[:g](k_n, k_o)

    bp = w - k_n - q*k_o - d
    
    ϕ_agent = 0
 τ_n = 1 .- (Par.β*(z*Fun[:fk](k)*Fun[:gn](k_n,k_o)*m .+ (1-Par.δ_n*(1-q))*(1 .+(1-Par.ρ)*ξp)) .+ Par.β*Par.θ*ϕ_agent*(1-Par.δ_n*(1-q))) ./ (1 .+ξ)

    τ_u = 1 .- (Par.β*(z*Fun[:go](k_n,k_o)*Fun[:fk](k)*m .+ (1-Par.δ_u)*q*(1 .+(1-Par.ρ)*ξp.+Par.θ*ϕ_agent))) ./ (q*(1 .+ξ))

    zz[1] = -(1 +ξ)*(1-τ) .+ Par.β*(z*Fun[:fk](k)*Fun[:gn](k_n,k_o)*m .+ (1-Par.δ_n*(1-q))*(1 .+(1-Par.ρ)*ξp)) .+ Par.β*Par.θ*ϕ_agent*(1-Par.δ_n*(1-q)) .+ Par.β*Par.δ_n*η

    zz[2] = -q*(1 .+ξ) .+ Par.β*(z*Fun[:fk](k)*Fun[:go](k_n,k_o)*m .+ (1-Par.δ_u)*q*(1 .+(1-Par.ρ)*ξp.+Par.θ*ϕ_agent)) - η*(1-Par.β*(1-Par.δ_u))

   
    return zz, τ_n, τ_u, ϕ_agent, bp

end
