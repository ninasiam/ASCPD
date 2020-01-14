function [L, beta, lambda_prox] = NAG_parameters_prox(Hessian, ops)
    
    Eigs = svd(Hessian);
    L = max(Eigs);
    mu = min(Eigs);
    ratio_var = ops.ratio_var;
    
    if (L/(mu + 10^(-6))) > 10^(5) && strcmp(ops.proximal,'true')
        lambda_prox = L/ratio_var;
    else
        lambda_prox = 0;
    end
    Q = (mu + lambda_prox)/(L + lambda_prox); 
    beta = ((1-sqrt(Q))/(1 + sqrt(Q)));
    
end
