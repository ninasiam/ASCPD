%FISTA VS ADMM
%Compare the problem for the matrix least squares problem

N = 50;                         %N is the dimension of the factor
M = 2500;                       %product of other dimensions
D = 10;                         %consider Rank
Y = randn(N,M);                 %matricization
X = randn(M,D);                 %the other factor (Khatri - Rao)
G_init = randn(N,D);            %current factor

P = X' * X;                     %Gram matrix
L = max(svd(P));                %L variable
q = Y*X;                     
eta = 1/L;
lambda_f = 0.1;
tol_FISTA = 10^(-5);

%Initializations for FISTA
G_fista = G_init;
iters = 1;
l = 0;
grad_f = - q - G_init*P;        %Gradient in terms of matrix X
Y_fista = G_fista;

while (1)                       %Fista main loop
    
    g_fista_old = G_fista;
    y_fista_old = Y_fista;
    l_old = l;                  %l parameter
    
    l = 1 + sqrt(1+4*l^2)/2;
    gamma = (1-l_old)/l;
    
    tmp = G_fista + eta * grad_f;
    Y_fista = soft_thresh(tmp, eta * lambda_f);           %soft-thresholding operator
    G_fista = (1-gamma)*Y_fista + gamma * y_fista_old; 
   
    if (norm(G_fista - g_fista_old) < tol_FISTA || iters > 1000)
        break; 
    end
    
    grad_f = -q - G_fista*P;
    iters = iters + 1;
    
end

%return;

%ADMM

lambda = 20;
rho = 10000;
iters_ADMM = 0;

ll = lambda/rho;

Z = Y*X*inv(X'*X);                  %LS update of Z
Z_init = Z;
%G_prev = (N,D);
H = zeros(N,D);

while(1)
    
    iters_ADMM = iters_ADMM + 1;
   
    %Update G
    %G = (Y*X - rho*(Z-H))*inv(X'*X - rho*eye(D)); %unconstraint solution for the matrix G
    M = (Y*X) - rho*(Z-H);
    M_gram = M'*M;
    [D,L] = eig(M_gram);
    L = L^(-1/2);
    G = M*D*L*D';
    %G = G';
    %Update Z 
    Z = G + H;               
    Z = (Z>ll).*(Z-ll) + (Z<-ll).*(Z+ll);

    %Update Y = H
    H = H + G - Z; 
    
    if iters_ADMM > 1
        r_n = norm(G - G_prev);
        if (r_n < 10^(-5) || iters_ADMM > 1000)

            break;

        end
    end
    G_prev = G;
    
    %F_value(iters) = norm(X-W*S','fro')+lambda*sum(sum(abs(W)));
   
%     figure(1)
%     plot(F_value)
%     pause(0.00001)
%     F_value(end) 
    
end