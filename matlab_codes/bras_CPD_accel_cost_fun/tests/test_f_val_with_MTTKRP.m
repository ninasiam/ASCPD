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
X = {};                                                                    % Input tensor

F = 10;% [60 80 90];
iter_mttkrp = 10;                                                          % Number of MTTKRPs
I_vec = [100];                                                             % Tensor size
bs = 10*[5, 5, 5];                                                       % Number of fibers
num_trial=1;                                                               % Number of trials
noise = [0];
bottlenecks = 'off';
corr_var = 0.9;
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
            A{i} = randi(1,1,1)*(rand(I{i},F));
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
            Hinit{d} = rand( I{d}, F );
        end
        
        
        %% BrasCPD accel
        fprintf('\nSolution with BrasAccel...\n')
        ops.constraint{1} = 'nonnegative';
        ops.constraint{2} = 'nonnegative';
        ops.constraint{3} = 'nonnegative';
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
        ops.ratio_var = 10000;
        
        [ A_accel, error_accel ,f_value, f_val_A_est, f_val_Y] = BrasCPD_vol3(X_data_tl,ops);
        len = length(error_accel);   
        error_nina_accel{i1}(trial,:)= [error_accel zeros(1,len-iter_mttkrp)];
        RFEaccel{i1}(trial) = max(cpderr(A_gt,A_accel))
        
    end
end