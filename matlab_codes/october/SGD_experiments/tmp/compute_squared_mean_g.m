function [ expected_value_g_square,stoch_squared ] = compute_squared_mean_g( batch, w, A, b)
  
    stoch_squared = zeros(1, length(batch));
    for ii=1:length(batch)
        stoch_squared(ii) = norm(A(batch(ii),:)'*(A(batch(ii),:)*w - b(batch(ii))))^2;
    end
    expected_value_g_square = mean(stoch_squared);
end

