
cd(@__DIR__)
using Pkg
Pkg.activate(pwd())

Pkg.add([
    "CategoricalArrays",
    "StatsModels",
])

using Distributions
using GLM
using CovarianceMatrices
using Plots
using MixedModels
using DataFrames
using CategoricalArrays
using StatsModels

# Set plotting theme
theme(:ggplot2)

# Configure default plot settings
default(
    size = (800, 400),
    fontfamily = "Computer Modern",
    tickfontsize = 12,
    legendfontsize = 12,
    titlefontsize = 12,
    xlabelfontsize = 12,
    ylabelfontsize = 12,
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


##### Mixed linear model estimation function
function build_panel_dataset(gcb_df::DataFrame, lulc_df::DataFrame)

    base_df = select(
        gcb_df,
        :Year,
        Symbol("fossil emissions excluding carbonation"),
        Symbol("atmospheric growth"),
    )

    rename!(
        base_df,
        Symbol("fossil emissions excluding carbonation") => :Fossil,
        Symbol("atmospheric growth") => :Growth,
    )

    measure_cols = filter(col -> col != "Year", names(lulc_df))

    long_lulc = stack(
        lulc_df,
        measure_cols;
        variable_name = :definition,
        value_name = :LULC,
    )

    long_df = innerjoin(base_df, long_lulc, on = :Year)
    long_df.definition = categorical(string.(long_df.definition))
    long_df.t = long_df.Year .- minimum(long_df.Year)
    long_df.AF = long_df.Growth ./ (long_df.Fossil .+ long_df.LULC)

    return long_df
end

function fit_mixed_model(long_df::DataFrame)
    full_formula = @formula(AF ~ t + (1 + t | definition))
    model = LinearMixedModel(full_formula, long_df)
    fit!(model)
    return model
end

function extract_group_coefficients(model, long_df::DataFrame)
    fe_names = fixefnames(model)
    fe = fixef(model)

    intercept_idx = findfirst(name -> occursin("Intercept", name), fe_names)
    slope_idx = findfirst(==("t"), fe_names)

    mu_alpha = fe[intercept_idx]
    mu_beta = fe[slope_idx]

    re = ranef(model)[1]
    defs = levels(long_df.definition)

    if size(re, 1) != 2 || size(re, 2) != length(defs)
        error("Unexpected random-effects shape. Expected 2 x number_of_definitions.")
    end

    group_df = DataFrame(
        definition = defs,
        alpha_j = mu_alpha .+ vec(re[1, :]),
        beta_j = mu_beta .+ vec(re[2, :]),
    )

    return group_df
end

function summarize_population_effects(model, group_df::DataFrame)
    fe_names = fixefnames(model)
    fe = fixef(model)
    fe_se = stderror(model)

    slope_idx = findfirst(==("t"), fe_names)
    intercept_idx = findfirst(name -> occursin("Intercept", name), fe_names)

    mu_alpha = fe[intercept_idx]
    mu_beta = fe[slope_idx]
    se_mu_beta = fe_se[slope_idx]

    z_mu_beta = mu_beta / se_mu_beta
    p_mu_beta = 2 * (1 - cdf(Normal(), abs(z_mu_beta)))

    sigma_alpha_hat = std(group_df.alpha_j)
    sigma_beta_hat = std(group_df.beta_j)

    summary_df = DataFrame(
        N_obs = nobs(model),
        N_definitions = length(unique(group_df.definition)),
        mu_alpha = mu_alpha,
        mu_beta = mu_beta,
        se_mu_beta = se_mu_beta,
        z_mu_beta = z_mu_beta,
        p_mu_beta = p_mu_beta,
        sigma_alpha_hat = sigma_alpha_hat,
        sigma_beta_hat = sigma_beta_hat,
    )

    return summary_df
end

function mixed_model_r2(model::LinearMixedModel)
    # Variance decomposition based on fixed-only and conditional fitted values.
    β = fixef(model)
    X = model.X

    yhat_fixed = X * β
    yhat_cond = fitted(model)

    var_fixed = var(yhat_fixed)
    var_random = max(var(yhat_cond .- yhat_fixed), 0.0)
    var_resid = model.sigma^2

    var_total = var_fixed + var_random + var_resid
    if var_total <= 0
        return (marginal = NaN, conditional = NaN)
    end

    R2_marginal = var_fixed / var_total
    R2_conditional = (var_fixed + var_random) / var_total

    return (marginal = R2_marginal, conditional = R2_conditional)
end


function extract_fixed_effect_stats(model, df::DataFrame; time_name::String="t", response_col::Symbol=:AF)
    names = fixefnames(model)
    betas = fixef(model)
    ses = stderror(model)

    idx_intercept = findfirst(name -> occursin("Intercept", name), names)
    idx_slope = findfirst(==(time_name), names)

    est_intercept = betas[idx_intercept]
    se_intercept = ses[idx_intercept]
    p_intercept = 2 * (1 - cdf(Normal(), abs(est_intercept / se_intercept)))

    est_slope = betas[idx_slope]
    se_slope = ses[idx_slope]
    p_slope = 2 * (1 - cdf(Normal(), abs(est_slope / se_slope)))

    r2 = mixed_model_r2(model)

    return (
        est_intercept = est_intercept,
        se_intercept = se_intercept,
        p_intercept = p_intercept,
        est_slope = est_slope,
        se_slope = se_slope,
        p_slope = p_slope,
        r2_marginal = r2.marginal,
        r2_conditional = r2.conditional,
    )
end

function build_mixed_fixed_effects_summary_table(model_full, df_full::DataFrame, model_sub, df_sub::DataFrame)
    stats_full = extract_fixed_effect_stats(model_full, df_full)
    stats_sub = extract_fixed_effect_stats(model_sub, df_sub)

    return DataFrame(
        Metric = ["Estimate", "Standard error", "p-value", "R-squared (marginal)", "R-squared (conditional)"],
        Intercept_full = [
            stats_full.est_intercept,
            stats_full.se_intercept,
            stats_full.p_intercept,
            stats_full.r2_marginal,
            stats_full.r2_conditional,
        ],
        Slope_full = [
            stats_full.est_slope,
            stats_full.se_slope,
            stats_full.p_slope,
            stats_full.r2_marginal,
            stats_full.r2_conditional,
        ],
        Intercept_up_to_2023 = [
            stats_sub.est_intercept,
            stats_sub.se_intercept,
            stats_sub.p_intercept,
            stats_sub.r2_marginal,
            stats_sub.r2_conditional,
        ],
        Slope_up_to_2023 = [
            stats_sub.est_slope,
            stats_sub.se_slope,
            stats_sub.p_slope,
            stats_sub.r2_marginal,
            stats_sub.r2_conditional,
        ],
    )
end

function dataframe_to_markdown_table(df::DataFrame; digits::Int=6)
    fmt_num(x) = string(round(x; digits = digits))

    header = "| Metric | Intercept (full) | Slope (full) | Intercept (up to 2023) | Slope (up to 2023) |"
    sep = "|---|---:|---:|---:|---:|"

    rows = String[]
    for r in eachrow(df)
        push!(rows, "| $(r.Metric) | $(fmt_num(r.Intercept_full)) | $(fmt_num(r.Slope_full)) | $(fmt_num(r.Intercept_up_to_2023)) | $(fmt_num(r.Slope_up_to_2023)) |")
    end

    return join([header, sep, rows...], "\n") * "\n"
end