clear;
clc;
close all

% add paths (home)
% addpath('/home/nina/Documents/uni/codes/matlab/master/demo_vol1/functions')
% addpath('/home/nina/Documents/uni/nina_s/matlab_codes/December/myBras_functions');
% addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');

% add paths (local-dell)
addpath('/home/telecom/Desktop/nina/matlab_codes/functions');
addpath('/home/telecom/Desktop/nina/nina_s/matlab_codes/December/myBras_functions');
addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');

%% Initializations
order = 3;
I = 100;
J = 100;
K = 100;

dims = [I J K];

display('Run with...');
R = 50%randi([10 min(dims)-10],1,1)
scale = 2;%randi([10 15],1,1);                                             % parameter to control the blocksize
B = scale*[10 10 10]                                                       % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

MAX_OUTER_ITER = 30;                                                       % Max number of iterations (epochs)

corr_var = 0.92;
%% create true factors 
for ii = 1:order
    A_true{ii} = rand(dims(ii),R);
end

A_bot1 = A_true{1};
A_bot2 = A_true{2};
A_bot3 = A_true{3};



A_bot1(:,R) = sqrt(corr_var)*A_bot1(:,R-1) + sqrt(1 - corr_var)*randn(I,1);
%A_bot1(:,R - 1) = sqrt(corr_var)*A_bot1(:,R-i) + sqrt(1 - corr_var)*randn(I{1},1);
A_bot2(:,R) = sqrt(corr_var)*A_bot2(:,R-1) + sqrt(1 - corr_var)*randn(J,1);
%A_bot2(:,R - 1) = sqrt(corr_var)*A_bot2(:,R-i) + sqrt(1 - corr_var)*randn(I{2},1);
A_bot3(:,R) = sqrt(corr_var)*A_bot3(:,R-1) + sqrt(1 - corr_var)*randn(K,1);
%A_bot3(:,R - 1) = sqrt(corr_var)*A_bot3(:,R-i) + sqrt(1 - corr_var)*randn(I{3},1);


A_true{1} = A_bot1;
A_true{2} = A_bot2;
A_true{3} = A_bot3;

X = cpdgen(A_true);                                                        % True tensor

%% create noise
for ii = 1:order
    noise{ii} = 0.1*randn(dims(ii),R);
end

X_N = cpdgen(noise);

T = X + X_N;

%% create initial point
for jj = 1:order
    A_init{jj} = rand(dims(jj),R);
end

%BrasCPD 
display('BrasCPD...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 1.5;
options.dims = dims;
options.acceleration = 'off';
options.cyclical = 'off';
options.proximal = 'off';
[A_est2, MSE2, error2] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est2)

%BrasCPD accel
display('BrasCPD accel...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'off';
options.ratio_var = 0;
[A_est, MSE, error] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est)

%BrasCPD ratio 1
display('BrasCPD accel with proximal term ratio = 1...');
 
options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 1;

[A_est_r1, MSE_r1, error_r1] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r1)

%BrasCPD ratio 10
display('BrasCPD accel with proximal term ratio = 10...');
 
options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 10;

[A_est_r10, MSE_r10, error_r10] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r10)

%BrasCPD ratio 100
display('BrasCPD accel with proximal term ratio = 100...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 100;

[A_est_r100, MSE_r100, error_r100] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r100)

%BrasCPD ratio 1000
display('BrasCPD accel with proximal term ratio = 1000...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 1000;

[A_est_r1000, MSE_r1000, error_r1000] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r1000)

%BrasCPD ratio 5000
display('BrasCPD accel with proximal term ratio = 5000...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 5000;

[A_est_r5000, MSE_r5000, error_r5000] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r5000)

%BrasCPD ratio 10000
display('BrasCPD accel with proximal term ratio = 10000...');

options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'true';
options.ratio_var = 10000;

[A_est_r10000, MSE_r10000, error_r10000] = BrasCPD_vol2(T,options);
cpderr(A_true,A_est_r10000)

%% plot

fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
semilogy([0:(size(MSE,2)-1)],MSE,'->y','linewidth',1.5);hold on
semilogy([0:(size(MSE2,2)-1)],MSE2,'->b','linewidth',1.5); hold on
semilogy([0:(size(MSE_r1,2)-1)],MSE_r1,'->k','linewidth',1.5);hold on
semilogy([0:(size(MSE_r10,2)-1)],MSE_r10,'->m','linewidth',1.5); hold on
semilogy([0:(size(MSE_r100,2)-1)],MSE_r100,'->c','linewidth',1.5);hold on
semilogy([0:(size(MSE_r1000,2)-1)],MSE_r1000,'->g','linewidth',1.5); hold on;
semilogy([0:(size(MSE_r5000,2)-1)],MSE_r5000,'-xg','linewidth',1.5); hold on;
semilogy([0:(size(MSE_r10000,2)-1)],MSE_r10000,'->r','linewidth',1.5); 
legend('BrasCP accel','BrasCP', 'BrasCP ratio 1', 'BrasCP ratio 10', 'BrasCP ratio 100', 'BrasCP ratio 1000', 'BrasCP ratio 5000', 'BrasCP ratio 10000');
xlabel('no. of MTTKRP computed')
ylabel('MSE')
set(gca,'fontsize',14)
grid on
% file_name = ['/home/telecom/Desktop/nina/matlab_codes/figures_15_1_2020_brasCPD_prox' '/' num2str(I) '_' num2str(R) '_with_1_5']
% saveas(fig1,[file_name '.pdf']);

fig2 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);
semilogy([0:(size(error,2)-1)],error,'->y','linewidth',1.5);hold on
semilogy([0:(size(error2,2)-1)],error2,'->b','linewidth',1.5); hold on
semilogy([0:(size(error_r1,2)-1)],error_r1,'->k','linewidth',1.5);hold on
semilogy([0:(size(error_r10,2)-1)],error_r10,'->m','linewidth',1.5); hold on
semilogy([0:(size(error_r100,2)-1)],error_r100,'->c','linewidth',1.5);hold on
semilogy([0:(size(error_r1000,2)-1)],error_r1000,'->g','linewidth',1.5); hold on;
semilogy([0:(size(error_r5000,2)-1)],error_r5000,'-xg','linewidth',1.5); 
semilogy([0:(size(error_r10000,2)-1)],error_r10000,'->r','linewidth',1.5); 
legend('BrasCP accel','BrasCP', 'BrasCP ratio 1', 'BrasCP ratio 10', 'BrasCP ratio 100', 'BrasCP ratio 1000','BrasCP ratio 5000', 'BrasCP ratio 10000');
xlabel('no. of MTTKRP computed')
ylabel('error')
set(gca,'fontsize',14)
grid on
% file_name = ['/home/telecom/Desktop/nina/matlab_codes/figures_15_1_2020_brasCPD_prox' '/' num2str(I) '_' num2str(R) '_' 'error_with_1_5'];
% saveas(fig2,[file_name '.pdf']);