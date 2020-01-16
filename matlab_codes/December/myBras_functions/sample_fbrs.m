function [ idxs ] = sample_fbrs( n, dims, B_n )

    order = size(dims,2);
    idxs = zeros(B_n, order);
    
    for ii = [1:n-1, n+1:order]
        idxs(:,ii) = randi(dims(ii), B_n, 1);
    end
    idxs = idxs(:,[1:n-1, n+1:order]);
end

