
%plot for five different experiments   
iters_katp = zeros(5,1);
iters_svrgp = zeros(5,1);
iters_sgdp = zeros(5,1);
iters_asgdp = zeros(5,1);

time_katp = zeros(5,1);
time_svrgp = zeros(5,1);
time_sgdp = zeros(5,1);
time_asgdp = zeros(5,1);


    ii = 1 %for five disfferent set of dimensions
    
    iters_katp(ii) = mean(iters_katyusha);
    iters_svrgp(ii) = mean(iters_svrgp);

    iters_sgdp(ii) = mean(iters_sgdp);
    iters_asgdp(ii) = mean(iters_asgdp);

    time_katp(ii) = mean(time_katyusha);
    time_svrgp(ii) = mean(time_svrgp);

    time_sgdp(ii) = mean(time_sgdp);
    time_asgdp(ii) = mean(time_asgdp);
    
