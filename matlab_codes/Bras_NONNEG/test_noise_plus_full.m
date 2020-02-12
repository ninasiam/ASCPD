clear;
clc;
close all


% add paths
addpath functions
addpath myBras_functions
addpath tensor_toolbox
addpath tensorlab_2016-03-28
addpath full_AO_NTF

% Problem setup
X = {};                         % Input tensor

F = 100;% [60 80 90];
iter_mttkrp = 60;               % Number of MTTKRPs
I_vec = [00];                  % Tensor size
bs = 2*[10, 10, 10];            % Number of fibers
num_trial=1;                    % Number of trials
noise = [0];


tol_inner_1 = 10^(-2);
tol_inner_2 = 10^(-2);
tol_outer = 10^(-7);

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
        A{i} = randi(5,1,1)*(rand(I{i},F));
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
    
    %XX = tensor(X{i1});

    
    
    X_data = X_data_t + XX_N;
    X_data_tl = X_data_tl + X_N{i1};
    
    % Initialize the latent factors   
    for d = 1:3
        Hinit{d} = rand( I{d}, F );
    end

    %% Full AO-NTF
    fprintf('\nSolution with Nesterov optimal method (Accel) (Norm.)...\n')
    [error, A, B, C, AO_iters] = NALS_Nesterov(X_data_tl, F, Hinit{1}, Hinit{2}, Hinit{3}, tol_inner_1, tol_inner_2, tol_outer);
    %re(trial,1) = frob(X_data_tl - cpdgen({A, B, C})) / frob(X_data_tl);
    rfe(trial,1) = max( cpderr(A_gt, {A, B, C}) )
 
%     %% BrasCPD stepsize 1  
%     fprintf('\nSolution with BrasCPD b0 = 0.1...\n')
%     ops.constraint{1} = 'nonnegative';
%     ops.constraint{2} = 'nonnegative';
%     ops.constraint{3} = 'nonnegative';
%     ops.b0 = 0.1;
%     ops.n_mb = bs(1);
%     ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
%     ops.A_ini = Hinit;
%     ops.A_gt=A_gt; % use the ground truth value for MSE computation
%     ops.tol= eps^2;
%     ops.step_opt = 'off';
%     rng('default');
%     [ A_1, MSE_A_1 ,NRE_A_1,TIME_A_1] = BrasCPD(X_data,ops);
%     len = length(MSE_A_1);
%     MSE_Xiao1{i1}(trial,:)= [MSE_A_1 zeros(1,len-iter_mttkrp)];
%     NRE_Xiao1{i1}(trial,:)= [NRE_A_1 zeros(1,len-iter_mttkrp)];
%     TIME_Xiao1{i1}(trial,:)=[TIME_A_1 zeros(1,len-iter_mttkrp)];
%     RFE1{i1}(trial) = max(cpderr(A_gt,A_1))
%     
%    
%     %% BrasCPD stepsize 2
%     fprintf('\nSolution with BrasCPD b0 = 0.05...\n')
%     ops.constraint{1} = 'nonnegative';
%     ops.constraint{2} = 'nonnegative';
%     ops.constraint{3} = 'nonnegative';
%     ops.b0 = 0.05;
%     ops.n_mb = bs(1);
%     ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
%     ops.A_ini = Hinit;
%     ops.A_gt=A_gt; % use the ground truth value for MSE computation
%     ops.tol= eps^2;
%     ops.step_opt = 'off';
%     rng('default');
%     [ A_2, MSE_A_2 ,NRE_A_2,TIME_A_2] = BrasCPD(X_data,ops);
%     len = length(MSE_A_2);
%     MSE_Xiao2{i1}(trial,:)= [MSE_A_2 zeros(1,len-iter_mttkrp)];
%     NRE_Xiao2{i1}(trial,:)= [NRE_A_2 zeros(1,len-iter_mttkrp)];
%     TIME_Xiao2{i1}(trial,:)=[TIME_A_2 zeros(1,len-iter_mttkrp)];
%     RFE2{i1}(trial) = max(cpderr(A_gt,A_2))

    
    
    %% BrasCPD stepsize 3
    fprintf('\nSolution with BrasCPD with optimal step...\n')
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.b0 = 0.01;
    ops.n_mb = bs(1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    ops.step_opt = 'on';
    rng('default');
    [ A_3, MSE_A_3 ,NRE_A_3,TIME_A_3] = BrasCPD(X_data,ops);
    len = length(MSE_A_3);
    MSE_Xiao3{i1}(trial,:)= [MSE_A_3 zeros(1,len-iter_mttkrp)];
    NRE_Xiao3{i1}(trial,:)= [NRE_A_3 zeros(1,len-iter_mttkrp)];
    TIME_Xiao3{i1}(trial,:)=[TIME_A_3 zeros(1,len-iter_mttkrp)];
    RFE3{i1}(trial) = max(cpderr(A_gt,A_3))

    %% BrasCPD accel
    fprintf('\nSolution with BrasAccel...\n')
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.bz = bs(1);
    ops.dims = dims;
    ops.max_iter = (I{1}*I{2}/ops.bz)*iter_mttkrp;
    ops.A_init = Hinit;
    ops.A_true=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    ops.alpha0 = 0.1;
    ops.tol_rel = 10^(-7);
    ops.acceleration = 'on';
    ops.cyclical = 'off';
    ops.proximal = 'true';
    ops.ratio_var = 1000;
    rng('default');
    [ A_accel, error_accel ,f_value] = BrasCPD_vol3(X_data_tl,ops);
%    len = length(MSE_A_accel);   
%     MSE_nina_accel{i1}(trial,:) = [MSE_A_accel zeros(1,len-iter_mttkrp)];
%     NRE_nina_accel{i1}(trial,:)= [NRE_A_accel zeros(1,len-iter_mttkrp)];
%     TIME_nina_accel{i1}(trial,:)= [TIME_A_accel zeros(1,len-iter_mttkrp)];
    RFEaccel{i1}(trial) = max(cpderr(A_gt,A_accel))

    
    %% AdaCPD
    fprintf('\nSolution with AdaCPD...\n')
    ops.constraint{1} = 'nonnegative';
    ops.constraint{2} = 'nonnegative';
    ops.constraint{3} = 'nonnegative';
    ops.eta = 1;
    ops.b0 = 1;
    ops.n_mb = bs(1);
    ops.max_it = (I{1}*I{2}/ops.n_mb)*iter_mttkrp;
    ops.A_ini = Hinit;
    ops.A_gt=A_gt; % use the ground truth value for MSE computation
    ops.tol= eps^2;
    rng('default');
    [ A_ada, MSE_A_adagrad ,NRE_A_adagrad, TIME_A_adagrad] = AdaCPD(X_data,ops);
    len = length(MSE_A_adagrad);   
    MSE_Xiao_adagrad{i1}(trial,:) = [MSE_A_adagrad zeros(1,len-iter_mttkrp)];
    NRE_Xiao_adagrad{i1}(trial,:)= [NRE_A_adagrad zeros(1,len-iter_mttkrp)];
    TIME_Xiao_adagrad{i1}(trial,:)= [TIME_A_adagrad zeros(1,len-iter_mttkrp)];
    RFEada{i1}(trial) = max(cpderr(A_gt,A_ada))

    
end

    %% plot
%     fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
%     semilogy([0:(size(MSE_Xiao1{i1},2)-1)],mean(MSE_Xiao1{i1},1),'-sb','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_Xiao2{i1},2)-1)],mean(MSE_Xiao2{i1},1),'-ob','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_Xiao3{i1},2)-1)],mean(MSE_Xiao3{i1},1),'->b','linewidth',1.5);hold on
%     semilogy([0:(size(MSE_nina_accel{i1},2)-1)],mean(MSE_nina_accel{i1},1),'-xy','linewidth',1.5);hold on
% 
%     semilogy([0:(size(MSE_Xiao_adagrad{i1},2)-1)],mean(MSE_Xiao_adagrad{i1},1),'-dg','linewidth',1.5);hold on
%     legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','BrasCP accel','AdaCPD')
%     xlabel('no. of MTTKRP computed')
%     ylabel('MSE')
%     set(gca,'fontsize',14)
%     grid on
%     file_name = ['noise' '_' num2str(SNR(i1))];
%     saveas(fig1,[file_name '.pdf']);
%     
%     fig2 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
%     semilogy([0:(size(NRE_Xiao1{i1},2)-1)],mean(NRE_Xiao1{i1},1),'-sb','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_Xiao2{i1},2)-1)],mean(NRE_Xiao2{i1},1),'-ob','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_Xiao3{i1},2)-1)],mean(NRE_Xiao3{i1},1),'->b','linewidth',1.5);hold on
%     semilogy([0:(size(NRE_nina_accel{i1},2)-1)],mean(NRE_nina_accel{i1},1),'-xy','linewidth',1.5);hold on
%     
%     semilogy([0:(size(NRE_Xiao_adagrad{i1},2)-1)],mean(NRE_Xiao_adagrad{i1},1),'-dg','linewidth',1.5);hold on
%     legend('BrasCPD (\alpha = 0.1)','BrasCPD (\alpha = 0.05)','BrasCPD (\alpha = 0.01)','BrasCP accel','AdaCPD')
%     xlabel('no. of MTTKRP computed')
%     ylabel('Relative Cost')
%     set(gca,'fontsize',14)
%     grid on
%     file_name = ['noise' '_' num2str(SNR(i1)) '_' 'cost'];
%     saveas(fig2,[file_name '.pdf']);
end    
     
     
figure(1)
semilogy(error);
hold on;
semilogy(NRE_A_3);
hold on;
semilogy(error_accel);
hold on;
semilogy(NRE_A_adagrad);
grid on;
legend('AO_NTF', 'BrasCPD optimal step', 'Brasaccel', 'Adagrad')



