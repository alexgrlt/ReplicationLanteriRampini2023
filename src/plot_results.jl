"""
This function plots the results of the model if one wants.
"""
function plot_results()

    (w_grid, kN_fb,kU_fb,k_fb) = run_model()

    l = @layout [a;b ;  c]


    p1 = plot(w_grid, kN_fb[:,1], linewidth=2, label = "Low-state")
    plot!(p1,w_grid, kN_fb[:,2], linewidth=2, label = "High-state")
    xlabel!(p1,"w")
    ylabel!(p1,"kN")


    p2 = plot(w_grid, kU_fb[:,1], linewidth=2,label = "Low-state")
    plot!(p2,w_grid, kU_fb[:,2], linewidth=2,label = "High-state")
    xlabel!(p2,"w")
    ylabel!(p2,"kU")


    p3 = plot(w_grid, k_fb[:,1], linewidth=2, label = "Low-state")
    plot!(p3,w_grid, k_fb[:,2], linewidth=2, label = "High-state")
    xlabel!(p3,"w")
    ylabel!(p3,"k")


    plot(p1,p2,p3,layout = l)
end