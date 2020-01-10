function [L, beta] = NAG_parameters(Hessian,lambda_prox)
    
    Eigs = svd(Hessian);
    L = max(Eigs);
    mu = min(Eigs);
    
    Q = (mu + lambda_prox)/(L + lambda_prox); 
    beta = ((1-sqrt(Q))/(1 + sqrt(Q)));
end

