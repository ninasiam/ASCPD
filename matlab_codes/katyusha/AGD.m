%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Nesterov Algorithm for a-stongly convex           %
%                     L-smooth fuctions                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Problem parameters
n = 50;
m = 70;

A = rand(m,n);
P = A'*A;
q = rand(n,1);

condition_n = cond(P);

eigs = eig(P);
max_eig = max(eigs);
min_eig = min(eigs);


%Initializations
x = zeros(n,1);
y = x;

L = max_eig;
mu = min_eig;

Q = inv(condition_n);
k = 1;

a = [];
a(1) = 0.3;
b = [];
b(1) = 0;

%Nesterov main loop (scheme II and scheme III)
while(1)
    
    if(k > 1250)
        break;
    end
    
    x(:,k+1) = y(:,k) - (1/L)*(P*y(:,k) + q);
    
    %for scheme II
    c = [1 (a(k)^2 - Q) -a(k)^2];
    s = roots(c);
    
    a(k+1) = s(s > 0); 
    
    b(k+1) = (a(k)*(1 - a(k)))/(a(k)^2 + a(k+1));
    
    y(:,k+1) = x(:,k+1) + b(k+1)*(x(:,k+1) - x(:,k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %for scheme III
    %y(:,k+1) = x(:,k+1) + ( (sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu)))*(x(:,k+1) - x(:,k));
    
    k = k + 1;
    
end
x_star = -inv(P)*q;
figure(1);
stem([x(:,k) - x_star]);
