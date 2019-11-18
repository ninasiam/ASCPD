function [ eta, alert_counter ] = check_error( error, window_length, eta, scaling_parameter, alert_counter)
%CHECK_ERROR Summary of this function goes here
%   Detailed explanation goes here
    lower_tol = 0.99;
    thresehold = 1;

    MAX_ITER = 2*100;
    
    left_window = norm( error(end - window_length + 1: end - window_length + floor(window_length/2)) );
    right_window = norm( error(end-window_length + floor(window_length/2) + 1 : end) );
    
    if right_window/left_window >= lower_tol
    %if error(end)/error(end - 1) <= tol
%         if mean(error(end - window_length + 1:end))/max(error(end - window_length + 1:end)) < thresehold
            % oscillations
            if alert_counter > MAX_ITER  
                fprintf('iter = %d',length(error));
                fprintf('\n---');
                eta = eta*(1-scaling_parameter);
                alert_counter = 0;
            else
                eta = eta;
                alert_counter = alert_counter + 1;
            end
%         else
%             % plateau
%             if alert_counter > MAX_ITER  
%                 fprintf('iter = %d',length(error));
%                 eta = eta*(1+scaling_parameter)
%                 fprintf('\n+++');
%                 alert_counter = 0;
%             else
%                 eta = eta;
%                 alert_counter = alert_counter + 1;
%             end  
%         end
    else
        eta = eta;
        alert_counter = 0;
    end
    
    if eta <= 10^(-5) 
        eta = 2*10^(-3);
    end
    
end

