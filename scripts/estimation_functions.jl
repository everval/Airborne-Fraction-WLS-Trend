
cd(@__DIR__)
using Pkg
Pkg.activate(pwd())

using Distributions
using GLM
using CovarianceMatrices
using Plots


# Set plotting theme
theme(:ggplot2)

# Configure default plot settings
default(
    size = (600, 400),
    fontfamily = "Computer Modern",
    tickfontsize = 10,
    legendfontsize = 12,
    titlefontsize = 12,
    xlabelfontsize = 10,
    ylabelfontsize = 10,
    titlefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    tickfontfamily = "Computer Modern"
)

function robust_est(Y, X; w=nothing, verbose=false)
    fit = isnothing(w) ? lm(X, Y) : lm(X, Y, wts = w)
    params = coef(fit)
    res = residuals(fit)
    rss = sum(res .^ 2)
    σ² = dispersion(fit)#rss / (length(Y) - size(X, 2))
    std_fit = stderror(fit)
    std_hac_fit = stderror(Bartlett{Andrews}(), fit)
    conf_int = [params .- 1.96*std_fit params .+ 1.96*std_fit]
    conf_int_hac = [params .- 1.96*std_hac_fit params .+ 1.96*std_hac_fit]
    t_st_hac = params ./ std_hac_fit    
    t_st = params ./ std_fit
    pvalues_hac = 2 .* (1 .- cdf.(Normal(), abs.(t_st_hac)))
    pvalues = 2 .* (1 .- cdf.(Normal(), abs.(t_st)))

    if verbose
        println("Estimated coefficients: ", params)
        println("Standard errors: ", std_fit)
        println("HAC Standard errors: ", std_hac_fit)
        println("t-statistics: ", t_st)
        println("HAC t-statistics: ", t_st_hac)
        println("p-values: ", pvalues)
        println("HAC p-values: ", pvalues_hac)
        println("Residual sum of squares: ", rss)
        println("σ²: ", σ²)
    end

    return (β = params, σ² = σ², stderr = std_fit, stderr_hac = std_hac_fit, 
        t_stat = t_st, t_stat_hac = t_st_hac, pvalues = pvalues, pvalues_hac = pvalues_hac, 
        u=residuals(fit), Yfit = GLM.predict(fit), betavardiag = std_fit.^2, 
        betavardiag_hac = std_hac_fit.^2, X = X, res=res, rsquared = r2(fit))
end
