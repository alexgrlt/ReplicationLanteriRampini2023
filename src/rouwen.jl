"""
Discretization method for a AR(1) process of the form ``z_{t+1} = \mu + \rho z_{t} + u_t`` where `` u_t \sim \mathcal{N}(0, \sigma)``
    rowen(N:: number nodes, μ:: mean of the process, σ:: sd of the innovation, ρ:: persistence)

Returns a tuple with the discretized grid for the realizations of the shock Z (a vector of size N), and its transition matrix (size NxN). 
"""
function rouwen(N,μ,σ,ρ)
    
    σz = σ / sqrt(1-ρ^2)

    p  = (1+ρ)/2
    Π = [p 1-p; 1-p p]

    for n = 3:N

        Π = p*[Π zeros(n-1,1); zeros(1,n)] + (1-p)*[zeros(n-1,1) Π; zeros(1,n)] + p*[zeros(1,n); zeros(n-1,1) Π]
        Π[2:end-1,:] = Π[2:end-1,:]/2
    
    end

    fi = sqrt(N-1)*σz
    Z  = collect(range(-fi,fi,length=  N))'
    Z  = Z .+ μ
    Z = vec(Z)

    return (Z,Π)
end