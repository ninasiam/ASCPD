clear;
clc;
close all

% add paths
addpath functions
addpath myBras_functions
addpath tensor_toolbox
addpath tensorlab_2016-03-28

% Problem setup
X = {};                                                                    % Input tensor

F = 60;% [60 80 90];
iter_mttkrp = 30;                                                          % Number of MTTKRPs
I_vec = [200];                                                             % Tensor size
bs = 20*[10, 10, 10];                                                      % Number of fibers
num_trial=1;                                                               % Number of trials
noise = [0];

bottlenecks = 'on';
corr_var = 0.9;
for i1 = 1:length(noise)
   
for trial = 1:num_trial
    
    
    disp('======================================================================================')
    disp(['running at trial ',num2str(trial), ': I equals ' ,num2str(I_vec(1)), ' and F equals ' ,num2str(F)])
    disp('======================================================================================')
    
    I{1} = I_vec(1);
    I{2} = I_vec(1);
    I{3} = I_vec(1);
    
    dims = [I_vec(1) I_vec(1) I_vec(1)];
        
    % Generate the true latent factors
    for i=1:3
        scale = randi(1,1,1)
        A{i} = scale*(randn(I{i},F));
    end
    
    
    if strcmp(bottlenecks,'on') 
        A_bot1 = A{1};
        A_bot2 = A{2};
        A_bot3 = A{3};



        A_bot1(:,F ) = sqrt(corr_var)*A_bot1(:,F - 1) + sqrt(1 - corr_var)*randn(I{1},1);
        %A_bot1(:,F - 2) = sqrt(corr_var)*A_bot1(:,F-1) + sqrt(1 - corr_var)*randn(I{1},1);
        A_bot2(:,F ) = sqrt(corr_var)*A_bot2(:,F - 1) + sqrt(1 - corr_var)*randn(I{2},1);
       % A_bot2(:,F - 2) = sqrt(corr_var)*A_bot2(:,F-1) + sqrt(1 - corr_var)*randn(I{2},1);
        A_bot3(:,F ) = sqrt(corr_var)*A_bot3(:,F - 1) + sqrt(1 - corr_var)*randn(I{3},1);
       % A_bot3(:,F - 2) = sqrt(corr_var)*A_bot3(:,F-1) + sqrt(1 - corr_var)*randn(I{3},1);


        A{1} = A_bot1;
        A{2} = A_bot2;
        A{3} = A_bot3;
    
    end
    
    A_gt = A;
    
    % Form the tensor
    for k=1:I{3}
        X{i1}(:,:,k)=A{1}*diag(A{3}(k,:))*A{2}';
    end
    X_data_t = tensor(X{i1});
    X_data_tl = cpdgen(A_gt);
    
    
    %Noise 
    for k=1:I{3}
        X_N{i1}(:,:,k)= noise(i1)*randn(I_vec, I_vec);
    end
    XX_N = tensor(X_N{i1});
    SNR(i1) = 10*log10(((1/I_vec^3)*norm(X_data_t)^2)/(noise(i1))^2)
    
    X_data = X_data_t + XX_N;
    X_data_tl = X_data_tl + X_N{i1};
    
    % Initialize the latent factors   
    for d = 1:3
        Hinit{d} = randn( I{d}, F );
    end
    
    %% BrasCPD step optimal
    fprintf('\nSolution with BrasCPD with optimal step...\n')
    ops.constraint{1} = 'noconstraint';
    ops.constraint{2} = 'noconstraint';
    ops.constraint{3} = 'noconstraint';
    ops.b0 = 0.01;
    ops.n_mb = bs(1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt;                                                         % use the ground truth value for MSE computation
    ops.tol= eps^2;
    ops.step_opt = 'on';
    rng('default');
    [ A_opt, MSE_A_opt ,error_opt, TIME_A_opt] = BrasCPD(X_data,ops);
    len = length(MSE_A_opt);
    MSE_opt{i1}(trial,:)= [MSE_A_opt zeros(1,len-iter_mttkrp)];
%    error_opt{i1}(trial,:)= [error_opt zeros(1,len-iter_mttkrp)];
    TIME_opt{i1}(trial,:)=[TIME_A_opt zeros(1,len-iter_mttkrp)];
    RFE_opt{i1}(trial) = max(cpderr(A_gt,A_opt))

    %% BrasCPD accel
    fprintf('\nSolution with BrasAccel...\n')
    ops.constraint{1} = 'noconstraint';
    ops.constraint{2} = 'noconstraint';
    ops.constraint{3} = 'noconstraint';
    ops.bz = bs(1);
    ops.dims = dims;
    ops.max_iter = (I{1}*I{2}/ops.bz)*iter_mttkrp;
    ops.A_init = Hinit;
    ops.A_true = A_gt;                                                     % use the ground truth value for MSE computation
    ops.tol= eps^2;
    ops.alpha0 = 0.1;
    ops.tol_rel = 10^(-7);
    ops.acceleration = 'on';
    ops.cyclical = 'off';
    ops.proximal = 'true';
    ops.ratio_var = 10;
    rng('default');
    [ A_accel, error_accel ,f_value] = BrasCPD_vol3(X_data_tl,ops);
%    len = length(MSE_A_accel);   
%     MSE_nina_accel{i1}(trial,:) = [MSE_A_accel zeros(1,len-iter_mttkrp)];
%     NRE_nina_accel{i1}(trial,:)= [NRE_A_accel zeros(1,len-iter_mttkrp)];
%     TIME_nina_accel{i1}(trial,:)= [TIME_A_accel zeros(1,len-iter_mttkrp)];
    RFEaccel{i1}(trial) = max(cpderr(A_gt,A_accel))

    
    %% AdaCPD
    fprintf('\nSolution with AdaCPD...\n')
    ops.constraint{1} = 'noconstraint';
    ops.constraint{2} = 'noconstraint';
    ops.constraint{3} = 'noconstraint';
    ops.eta = 1;
    ops.b0 = 1;
    ops.n_mb = bs(1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt = A_gt;                                                       % use the ground truth value for MSE computation
    ops.tol= eps^2;
    rng('default');
    [ A_ada, MSE_A_adagrad ,error_A_adagrad, TIME_A_adagrad] = AdaCPD(X_data,ops);
    len = length(MSE_A_adagrad);   
    MSE_adagrad{i1}(trial,:) = [MSE_A_adagrad zeros(1,len-iter_mttkrp)];
%    error_adagrad{i1}(trial,:)= [error_A_adagrad zeros(1,len-iter_mttkrp)];
    TIME_adagrad{i1}(trial,:)= [TIME_A_adagrad zeros(1,len-iter_mttkrp)];
    RFEada{i1}(trial) = max(cpderr(A_gt,A_ada))

    
end
    %% CPD_RBS
     [A_est_rbs, output] = cpd_rbs(X_data_tl,Hinit,'BlockSize', [10 10 10]);
     RFE_rbs(trial) = max(cpderr(A_gt,A_est_rbs))
     
     
    %% plot
    
    %print_results();

end    
     
     
figure(1)
semilogy(error_opt);
hold on;
semilogy(error_accel);
hold on;
semilogy(error_A_adagrad);
grid on;
legend('BrasCPD optimal step', 'Brasaccel', 'Adagrad')



