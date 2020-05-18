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

F = 50;% [60 80 90];
iter_mttkrp = 21;                                                          % Number of MTTKRPs
I_vec = [200];                                                             % Tensor size
bs = 5*[10, 10, 10];                                                      % Number of fibers
num_trial=3;                                                               % Number of trials
noise = [0 0.01];

bottlenecks = 'off';
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

        %% BrasCPD  step = 0.1
        fprintf('\nSolution with BrasCPD with alpha = 0.1 ...\n')
        ops.constraint{1} = 'noconstraint';
        ops.constraint{2} = 'noconstraint';
        ops.constraint{3} = 'noconstraint';
        ops.b0 = 0.1;
        ops.n_mb = bs(1);
        ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
        ops.A_ini = Hinit;
        ops.A_gt = A_gt;                                                   % use the ground truth value for MSE computation
        ops.tol= eps^2;
        ops.step_opt = 'off';
        
        [ A_1, MSE_A_1 ,error_A_1,TIME_A_1] = BrasCPD(X_data,ops);
        len = length(error_A_1);
        error_1{i1}(trial,:)= [error_A_1 zeros(1,len-iter_mttkrp)];
        RFE1{i1}(trial) = max(cpderr(A_gt,A_1))
        
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
        [ A_opt, MSE_A_opt ,error_A_opt, TIME_A_opt] = BrasCPD(X_data,ops);
        len = length(error_A_opt);
        error_opt{i1}(trial,:)= [error_A_opt zeros(1,len-iter_mttkrp)];
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
        ops.ratio_var = 5;
        [ A_accel, error_A_accel ,f_value, f_val_A, f_val_Y] = BrasCPD_vol3(X_data_tl,ops);
        len = length(error_A_accel);
        error_nina_accel{i1}(trial,:)= [error_A_accel zeros(1,len-iter_mttkrp)];
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
        [ A_ada, MSE_A_adagrad ,error_A_adagrad, TIME_A_adagrad] = AdaCPD(X_data,ops);
        len = length(error_A_adagrad);   
        error_adagrad{i1}(trial,:) = [error_A_adagrad zeros(1,len-iter_mttkrp)];
        RFEada{i1}(trial) = max(cpderr(A_gt,A_ada))

        %% CPD_RBS
        bs_rbs = [ceil(sqrt(I{1}*I{2})) ceil(sqrt(I{1}*I{2})) ceil(sqrt(I{1}*I{2}))];
        [A_est_rbs, output] = cpd_rbs(X_data_tl,Hinit,'BlockSize', bs_rbs,'MaxIter',8);
        error_rbs{i1}(trial,:) = output.error;
        RFE_rbs{i1}(trial) = max(cpderr(A_gt,A_est_rbs))
    end
    
    %% plot 
    input = struct;
    input.ratio = ops.ratio_var;
    input.constraint = 'unconstraint';
    input.path = ['dims_' num2str(I{1}) '_F_' num2str(F) '_bs_' num2str(bs(1)) '_proximal_' ops.proximal '_rv_' num2str(ops.ratio_var) ''];
    input.cpdrbs = error_rbs{i1};
    input.bras_opt = error_opt{i1};
    input.bras_1 = error_1{i1};
    input.bras_accel = error_nina_accel{i1};
    input.ada = error_adagrad{i1};
    res = print_results(SNR(i1), input,bottlenecks);
end    

t = count_percentage(trial, error_rbs{i1}, error_opt{i1}, error_nina_accel{i1}, error_adagrad{i1});

percentage = t/trial;
     
% figure(1)
% semilogy(error_opt);
% hold on;
% semilogy(error_accel);
% hold on;
% semilogy(error_A_adagrad);
% grid on;
% legend('BrasCPD optimal step', 'Brasaccel', 'Adagrad')
% 
% figure(2)
% semilogy(f_val_A);
% hold on;
% semilogy(f_val_Y);
% grid on;
% legend('seq A_est', 'seq Y')
% title('F value at iter')

