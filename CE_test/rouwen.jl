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