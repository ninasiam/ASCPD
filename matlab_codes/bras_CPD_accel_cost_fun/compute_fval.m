function [f_val] = compute_fval(frob_X, A, T)
% cost function value using the identity with MTTKRP
% in respect to first mode
    MTTKRP = tens2mat(T, 1, [2 3])*kr(A{3},A{2});
    Had_Grammian = (A{1}'*A{1}) .* (A{2}'*A{2})...         
               .* (A{3}'*A{3});
    f_val = sqrt(frob_X^2 + sum(sum(Had_Grammian)) -2*sum(sum(MTTKRP.*A{1})));

end

