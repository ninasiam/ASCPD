function [RFC] = rel_measure(A_est_next, A_est)
    
    %Relative Factor Change and error foe ach factor
    for ii = 1:size(A_est,2)
        RFC(ii) = frob(A_est_next{ii} - A_est{ii})/frob(A_est{ii});
    end
    
    RFC = max(RFC);
    
end

