function [t] = conut_percentage(trials,error_full, error_opt, error_nina_accel, error_adagrad)
t = 0;    
for ii=1:trials
        if(error_nina_accel(ii,end) > error_opt(ii,end) || ...
          error_nina_accel(ii,end) > error_adagrad(ii,end) )
             t = t + 1;
        end
    end
end

