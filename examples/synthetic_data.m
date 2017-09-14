%% Script which showcases how this structured VB NMF works on artificial data
%

% Width and height of the input matrix, which is to be decomposed 
W = 100;
K = 70;
% 'I' is the number of factors
I = 10;
fanin = 3; %max fan-in for the adhacency matrix

%% Adjacency matrix creation
%
adjacency = mk_rnd_dag(K,fanin);
adjacency = sparse(adjacency);
% topological ordering will be useful later
order = topological_sort(adjacency);
adjacency = adjacency(order, order);
% /adjacency

% csvwrite('adjacency.dat', full(adjacency), 0, 2)

% Hyperparameters of the underlying generative model
%
% matrix V -> the right one
numparents = sum(adjacency);
noparents = numparents==0;
nr_params_to_initialize = sum(noparents);
noparents = find(noparents);
a_ve = ones(I,K)*100;
mean_ve = 0 + (100-0).*rand(I,K);
b_ve = mean_ve./a_ve;
% mean is a_ve*b_ve
%
% matrix T -> the left one. Let's draw its elements from an exponential distro
a_tm = ones(W,I);
b_tm = ones(W,I); % rightfactor will be mean of exponential distros
%
% /hyperparameters

% Finally generating input data, sampling from the generative model
%
% having defined the free parameters of the generative model, we can
% proceed to sampling
V = zeros(I,K);
V(:,noparents) = gamrnd(a_ve(:,noparents), b_ve(:,noparents));
%
% now topological ordering comes in handy - because when we sample from a
% distribution of a node, we have to be sure that we sampled from all its
% ancestors and with a topologically ordered adjacency matrix we can
% sample from the first listed node to last with this condition satisfied
relevant_indices = find( numparents>=1 );
for(current = relevant_indices)
    % a safe way to get the current's parents:
    parents = find( adjacency(:,current) );
    active_parent = parents(1);
    % a_ve are propagated from the topmost parent to all its children
    V(:,current) = gamrnd(a_ve(:,current), V(:,active_parent)./a_ve(:,current));
end
T = gamrnd(a_tm, b_tm);
%
% T and V are ground truth, and X is the mixture
X = poissrnd(T*V);

Xtrue = X;
Ttrue = T;
Vtrue = V;

% Missing values are denoted by nans. Let's make some of the elements of X
% unobserved.
X(rand(size(X))>0.05) = NaN; % remove 95% of entries
