rng(0);
order = 3;
I = 5;
J = 5;
K = 5;

dims = [I J K];

display('Run with...');
R = 3%randi([10 min(dims)-10],1,1)
scale = 2;%randi([10 15],1,1);                                             % parameter to control the blocksize
B = scale*[1 1 1]                                                       % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

                                                       % Max number of iterations (epochs
%% create true factors 
for ii = 1:order
    A_true{ii} = rand(dims(ii),R);
end

T = cpdgen(A_true)

for k=1:K
    X(:,:,k)=A_true{1}*diag(A_true{3}(k,:))*A_true{2}';
end
X_data = tensor(X);

idxs = zeros(B(1), order);

n = 2;


for ii = [1:n-1, n+1:order]
    rng(1);
    idxs(:,ii) = randi(dims(ii), B(n), 1)
end


factor_idxs = idxs(:,[1:n-1, n+1:order])

if n == 1
for ii = 1:B(n)
    T_s(ii,:) = T(:,idxs(ii,2),idxs(ii,3));
end

elseif n == 2
for ii = 1:B(n)
    T_s(ii,:) = T(idxs(ii,1),:,idxs(ii,3));
end

elseif n == 3
for ii = 1:B(n)
    T_s(ii,:) = T(idxs(ii,1),idxs(ii,2),:);
end
end
[tensor_idx, factor_idx]  = sample_fibers2(B(n), dims, n)


X_sample = reshape(X_data(tensor_idx), dims(n), [])'


