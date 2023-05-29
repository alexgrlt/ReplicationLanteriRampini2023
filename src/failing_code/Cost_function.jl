function Cost(d,Par)

    # This function computes the cost of dividends $d$ for the firm and its first and second derivatives, given
    # the parameters of the model in the dictionary.

    C = zeros(length(d), 1)
    Cp = zeros(length(d), 1)
    Cpp = zeros(length(d), 1)

    for i = 1:length(d)
        
        if d[i] >= 0
        
        
            C[i] =0
            Cp[i] = 0
            Cpp[i] = 0
        
        
        else
            C[i] = Par.χ0 .*abs(d[i]).^Par.χ1
            Cp[i] = Par.χ0 .*Par.χ1 .*abs(d[i]).^(Par.χ1-1)
            Cpp[i] = Par.χ0 .* Par.χ1 .* (Par.χ1 -1).*abs(d[i]).^(Par.χ1 -2)
        
        end
    end

    return (C, Cp, Cpp)

end