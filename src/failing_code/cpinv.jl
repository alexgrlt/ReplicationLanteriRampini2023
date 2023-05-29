function Cpinv(ξ, Par)

    # inverse cost function (defined in Cost_function.jl) given ξ and parameters of the model

    d = 0
    if ξ > 0
        d = -(ξ/(Par.χ0 * Par.χ1))^(1/(Par.χ1 -1))
    end
    return d
end