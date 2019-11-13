function [ eta, alert_counter ] = check_error( error, window_length, eta, scaling_parameter, alert_counter)
%CHECK_ERROR Summary of this function goes here
%   Detailed explanation goes here
    tol = 0.80;
    MAX_ITER = 10;
    
    left_window = norm( error(end - window_length + 1: end - window_length + floor(window_length/2)) );
    right_window = norm( error(end-window_length + floor(window_length/2) + 1 : end) );
    
    if right_window/left_window >= tol || left_window/right_window >= tol
        if alert_counter > MAX_ITER  
            eta = eta / scaling_parameter;
            alert_counter = 0;
        else
            eta = eta;
            alert_counter = alert_counter + 1;
        end
    else
        eta = eta;
        alert_counter = 0;
    end

end

