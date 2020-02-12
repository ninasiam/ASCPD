function factor = select_factor(cyclical,iter,order,cyclical_vec)

    if strcmp('on',cyclical)
        if mod(iter,order) == 0
            n = cyclical_vec(end);
        elseif mod(iter,order) == 1
            n = cyclical_vec(end-1);
        elseif mod(iter,order) == 2
            n = cyclical_vec(end-2);
        end
    else
        n = randi(order,1);
    end
    
    factor = n;
    
end

