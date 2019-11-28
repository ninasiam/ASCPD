function [a_i_a_j, a_jx_bj] = check_correlations(A, b)
    
    a_i_a_j = A*A';
    y = A*randn(size(A,2),1) - b;
    a_jx_bj = y*y';
end

