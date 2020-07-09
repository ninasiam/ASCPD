%% Logistic Regression (GD/SGD/AgaGrad/AccelStoch)
%  Authors: Ioanna Siaminou, Giannis Papagiannakos, Chris Kolomvakis
%  date: 9/06/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
clear ; close all; clc

addpath('Libraries/cvx')

%% Create Data
m = 50;                                                                    % features
n = 2;                                                                     % parameter dimension
lambda = 1;                                                                % regularization parameter


%% Create features
% (fully separable)
% Ground truth 
% x_star = ones(n, 1);				
% c = 10*sign(randn);
% A = [c*ones(m/2,n); -c*ones(m/2,n)];
% A = A + 5*randn(m,n);
% threshold = 0;
% param = A*x_star > threshold;
% % Measurements in y
% y = param;

% (non-separable)
% Ground truth 
% x_star = (1/2)*randn(n, 1);
% A = 5*randn(m,n);
% % Measurements in y
% y = (rand(m,1) < exp(A*x_star)./(1+exp(A*x_star)));
% one_v = ones(m, 1);

% Data generation
x_star = 1 * randn(2,1);
mu1 = [-x_star(2); x_star(1)];
mu2 = -mu1;
for ii=1:m
     rnd=rand;
     if rnd>0.5
          x(:,ii) = mu1 +2*randn(2,1);
          if (x_star'*x(:,ii) > 0), y(ii)=1; else y(ii)=0; end
     else
          x(:,ii) = mu2 + 2*randn(2,1);
          if (x_star'*x(:,ii) > 0), y(ii)=1; else y(ii)=0; end
     end
end

A = x';
y = y';

%% CVX
% cvx_expert true
% cvx_begin
%     variables x_cvx(n)
%     minimize((1/m)*(-y'*A*x_cvx - sum(log_sum_exp(x_cvx'*A')) ))
% cvx_end

%% Plot data
% figure(1)
% fig1 = plot_data(A, y);
% hold off;

%% Initial values
x_init = 3*ones(n, 1);

% log loss
param = A*x_init;
sigm_x = (1./(1 + exp(-param)));
log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));

% initial cost function value
J_init = sum(log_loss)/m + (lambda/(2))*norm(x_init)^2;

%% Solution via Gradient
alpha = 0;                                                                 % alpha ~= 0 (constant step size)
fprintf('Solution via GD...\n');
[x_gd, J_gd, x_list1] = solve_GD(n, m, A, y, J_init, x_init, sigm_x, alpha, lambda, 200);

%% Solution via Stochastic Gradient
alpha = 0;
fprintf('Solution via SGD...\n');
[x_sgd, J_sgd, x_list2] = solve_SGD(n, m, A, y, J_init, x_init, sigm_x, alpha, lambda, 200*m);

%% Solution via Adagrad
alpha = 0.1;
fprintf('Solution via AdaGrad...\n');
[x_ada, J_ada, x_list3] = solve_AdaGrad(n, m, A, y, J_init, x_init, sigm_x, alpha, lambda, 200*m);

%% Solution via Stochastic (Accelerated)
alpha = 0;
fprintf('Solution via  Stochastic (Accelerated)...\n');
[x_asgd, J_asgd, x_list4] = solve_ASGD(n, m, A, y, J_init, x_init, sigm_x, alpha, lambda, 200*m);

%% Plot cost function
dt = 0.01;
x1 = -4:dt:4;                                                              % range of x1, x2
x2 = -4:dt:4;

[X1,X2] = meshgrid(x1,x2);

for i =1:length(x1)
    for j = 1:length(x2)
        x_val = [x1(i) x2(j)]';
        % log loss
        param = A*x_val;
        sigm_x = (1./(1 + exp(-param)));
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J(i,j) = sum(log_loss)/m + (lambda/(2))*norm(x_val)^2;
    end
end
figure()
mesh(X1,X2,J)                                                              % plot cost function (3D)

figure();
contour(X1, X2, J',100);
hold on;
plot(x_list1(1,:), x_list1(2,:), 'r-+')
hold on;
plot(x_list2(1,:), x_list2(2,:), 'g-+')
hold on;
plot(x_list3(1,:), x_list3(2,:), 'm-+')
hold on;
plot(x_list4(1,:), x_list4(2,:), 'y-+' )
legend('Contour lines', 'GD', 'SGD', 'AdaGrad', 'Stochastic gradient Accelerated')