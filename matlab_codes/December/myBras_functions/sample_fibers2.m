function [tensor_idx, factor_idx] = sample_fibers2(n_fibers, dim_vec, d)
    dim_len = length(dim_vec);
    tensor_idx = zeros(n_fibers, dim_len);

    % Randomly select fibers for dimensions not d
    for i = [1:d-1,d+1:dim_len]
        rng(1);
       tensor_idx(:,i) = randi(dim_vec(i), n_fibers, 1);
    end
    factor_idx = tensor_idx(:,[1:d-1,d+1:dim_len]);

    % Convert it into tensor indices
    tensor_idx = kron(tensor_idx,ones(dim_vec(d),1)); 
    tensor_idx(:,d) = repmat((1:dim_vec(d))',n_fibers,1);
end