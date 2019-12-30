function [L, beta] = NAG_parameters(Hessian)
    
    Eigs = svd(Hessian);
    L = max(Eigs);
    mu = min(Eigs);
    
    Q = mu/L;
    beta = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
end

