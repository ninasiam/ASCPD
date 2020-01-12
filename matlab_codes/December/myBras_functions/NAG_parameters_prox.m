function [L, beta, lambda_prox] = NAG_parameters_prox(Hessian, ops)
    
    Eigs = svd(Hessian);
    L = max(Eigs);
    mu = min(Eigs);

    if mu < 10^(-8) && strcmp(ops.proximal,'true')
        lambda_prox = L/1000;
    else
        lambda_prox = 0;
    end
    Q = (mu + lambda_prox)/(L + lambda_prox); 
    beta = ((1-sqrt(Q))/(1 + sqrt(Q)));
    
end
