function [ A_norm ] = normalize( A )
   
    [m,n] = size(A);
    for ii = 1:m
        A_norm(ii,:) = A(ii,:)./norm(A(ii,:));
    end

end

