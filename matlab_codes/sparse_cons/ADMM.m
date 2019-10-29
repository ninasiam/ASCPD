clc; clear all; 

N = 10000; 
M = 100;
X = randn(N,M);
R = 3;
lambda = 20;
rho = 10000;

S = (N*R)*randn(M,R);
iters = 0;

while(1)
    iters = iters + 1
    iter = 1;
    T = X*S;
    ll = lambda/rho;

    Y = zeros(N,R);
    [D,L] = eig(T'*T);
    L = diag(1./sqrt(diag(L)));
    Z = T*D*L*D';
    r = [];
    while(1)
        iter = iter+1;

        %Update W
        M = 2*T-rho*(Y-Z);
        [D,L] = eig(M'*M);
        L = diag(1./sqrt(diag(L)));
        W = M*D*L*D';

        %Update Z
        Z = W + Y;               
        Z = (Z>ll).*(Z-ll) + (Z<-ll).*(Z+ll);

        %Update Y
        Y = Y + W - Z;
        r = norm(W-Z,'fro');

        if (r<10^(-3)) || (iter > 10000)
            r
            if sum(sum(Z.^2)>0) == R
              W = Z;
            end
            break;
        end
    end

    S = X'*W;
     
    hold off
    pause(0.001)
    
    F_value(iters) = norm(X-W*S','fro')+lambda*sum(sum(abs(W)));
    
    figure(1)
    plot(F_value)
    pause(0.00001)
    F_value(end) 
    
end
    