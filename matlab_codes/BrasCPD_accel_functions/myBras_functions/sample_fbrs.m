function [ idxs, factor_idxs, T_s ] = sample_fbrs( T, n, dims, B_n )

    order = size(dims,2);
    idxs = zeros(B_n, order);
    
    for ii = [1:n-1, n+1:order]
        idxs(:,ii) = randi(dims(ii), B_n, 1);
    end
    %idxs = idxs(:,[1:n-1, n+1:order]);
    
    factor_idxs = idxs(:,[1:n-1, n+1:order]);

	if n == 1
        for ii = 1:B_n
            T_s(ii,:) = T(:,idxs(ii,2),idxs(ii,3));
        end

    elseif n == 2
        for ii = 1:B_n
            T_s(ii,:) = T(idxs(ii,1),:,idxs(ii,3));
        end

    elseif n == 3
        for ii = 1:B_n
            T_s(ii,:) = T(idxs(ii,1),idxs(ii,2),:);
        end
end

