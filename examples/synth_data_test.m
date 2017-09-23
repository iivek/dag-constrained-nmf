addpath( '../external-lib/bnt/graph' )
addpath( '../external-lib/bnt/KPMtools')
addpath( '../')

%% experimenting with the algorithm on synthetic data
clear all;
synthetic_data;

% initial parameters for the algorithm
% Width and height of the input matrix, which is to be decomposed 
[W,K] = size(X);
I = 5; % number of factors

% not interested in separate DAGs - fixing a single a_ve for each row
%a_ve = ones (I,1)*100;
a_ve = ones(I,nr_params_to_initialize)*2;
b_ve = 10./a_ve;
%b_ve = b_ve(:,noparents);
a_tm = ones(W,I);
b_tm = ones(W,I); % rightfactor will be mean of exponential distros

% Dirichlet - related parameters. Wherever we have number of parents i > 1,
% we'll need to put a vector of size i inside the corresponding cell 
% Note: other Dirichlet parameters, if defined, will be ignored.
u_dirichlet_init = cell(K,1);
relevant_indices = 1:K;
relevant_indices = relevant_indices(numparents>1);
for(i = relevant_indices)
    u_dirichlet_init{i} = ones(numparents(i),I).*0.5; 
end
u_dirichlet = u_dirichlet_init;
clear u_dirichlet_init;

[T_est V_est a_tm_est b_tm_est a_ve_est b_ve_est M] = structuredNMF_VB(X, a_tm, b_tm.*a_tm, a_ve, b_ve.*a_ve, 100, u_dirichlet, adjacency, ...
    500, ... % EPOCH
    10, ... % optimize hyperparameters after
    'semifree', ... % let scale be optimized, shsape is as-is 
    'clamp', ... % clamp shape, insist on sparsity by leaving scale = 1
    'tie_all', ... % optimize scale, which is unique for all elements of T
    500 ...    % print period    
);

X_est = T_est*V_est;
Xtrue-X_est


%% Okay... Now without the adjacency matrix
adjacency = zeros(size(adjacency));
numparents = sum(adjacency);
noparents = numparents==0;
nr_params_to_initialize = sum(noparents);
noparents = find(noparents);

% initial parameters for the algorithm
% Width and height of the input matrix, which is to be decomposed 
[W,K] = size(X);
I = 5; % number of factors

% not interested in separate DAGs - fixing a single a_ve for each row
%a_ve = ones (I,1)*100;
a_ve = ones(I,nr_params_to_initialize)*2;
b_ve = 10./a_ve;
%b_ve = b_ve(:,noparents);
a_tm = ones(W,I);
b_tm = ones(W,I); % rightfactor will be mean of exponential distros

% Dirichlet - related parameters. Wherever we have number of parents i > 1,
% we'll need to put a vector of size i inside the corresponding cell 
% Note: other Dirichlet parameters, if defined, will be ignored.
u_dirichlet_init = cell(K,1);
relevant_indices = 1:K;
relevant_indices = relevant_indices(numparents>1);
for(i = relevant_indices)
    u_dirichlet_init{i} = ones(numparents(i),I).*0.5; 
end
u_dirichlet = u_dirichlet_init;
clear u_dirichlet_init;

[T_est_no V_est_no a_tm_est b_tm_est a_ve_est b_ve_est M] = structuredNMF_VB(X, a_tm, b_tm.*a_tm, a_ve, b_ve.*a_ve, 1000, u_dirichlet, adjacency, ...
    500, ... % EPOCH
    10, ... % optimize hyperparameters after
    'semifree', ... % let scale be optimized, shsape is as-is 
    'clamp', ... % clamp shape, insist on sparsity by leaving scale = 1
    'tie_all', ... % optimize scale, which is unique for all elements of T
    500 ...    % print period    
);

X_est_no = T_est_no*V_est_no;
% Xtrue-X_est_no

% Some visualization
figure(1);
subplot(4,1,1); imagesc(Xtrue); title('Original data (oracle).'); colormap bone;
subplot(4,1,2); imagesc(X); title('Available data, used in reconstruction of the original data.'); colormap bone;
subplot(4,1,3); imagesc(X_est); title(strcat('Reconstruction with background knowledge included. mse=', num2str(mse(Xtrue-X_est),'%8.5e'))); colormap bone;
subplot(4,1,4); imagesc(X_est_no); title(strcat('Reconstruction ignorant of background knowledge. mse=', num2str(mse(Xtrue-X_est_no),'%8.5e'))); colormap bone;

figure(2)
subplot(3,1,1); imagesc(Vtrue./repmat(max(Vtrue,[],2),[1,size(Vtrue,2)])); title('Original latent factors. Corellations can be noted, between columns.'); colormap bone;
subplot(3,1,2); imagesc(V_est./repmat(max(V_est,[],2),[1,size(V_est,2)])); title('Reconstructed latent factors, with background knowledge included. Corellations should be present here, driven by background knowledge.'); colormap bone;
subplot(3,1,3); imagesc(V_est_no./repmat(max(V_est_no,[],2),[1,size(V_est_no,2)])); title('Reconstructed latent factors, no background knowledge.'); colormap bone;





