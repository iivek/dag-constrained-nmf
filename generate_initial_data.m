%% This is the initial part taken from structuredNMF_vb.

% Dimensions of matrices
W = size(x,1);
K = size(x,2);
I = size(b_tm,2);

M = ~isnan(x);
X = zeros(size(x));
X(M) = x(M);

[rows_mask cols_mask] = find(M);
rows_mask = uint32(rows_mask);
cols_mask = uint32(cols_mask);
linear_mask = find(M);
linear_mask = uint32(linear_mask);
x_raw = full(x(linear_mask));
clear x;

[gamma_chain_adjacency first_s] = conjugate_gamma_filler(adjacency);
last_v = first_s-1;
numparents = sum( gamma_chain_adjacency );
network_size = size( gamma_chain_adjacency,1 );


% Parameter initialization
%
expectations_discrete = cell(network_size,1);
expectations_dirichlet = cell(network_size,1);

indeces = find(numparents>1);
for(i = indeces)
    parent_indices = find( gamma_chain_adjacency(:,i) );
    expectations_discrete{i} = ones(numparents(i),I)./numparents(i);
end
clear initial;

relevant_indeces = 1:network_size;
relevant_indeces = relevant_indeces(numparents>1);
for(i = relevant_indeces)
    expectations_dirichlet{i} = psi(u_dirichlet{i}) - ...
        ones(size(u_dirichlet{i}))*diag( psi(sum(u_dirichlet{i},1)) );
end

% There will be one b_ve for each topmost element of Bayesian network.
% Others will not be stored, since they are calculated from their parent's
% parameters and expectations.
% Decision where to pull out b parameter of gamma from will be made from
% the query if the node has a parent, in the following way:
%   - if so, calculate b from the parents
%   - if not so, read b from b_ve

% Elements of v_e with free b_ve parameters, rowwise
where_we_have_b = find(sum(gamma_chain_adjacency) == false); % Note: bad style - treating bools as ints, but MATLAB can handle it.
% Note that a_ve and b_ve need not neccessarily be of same sizes
b_ve_sparse = sparse(I, network_size);
b_ve_sparse(:,where_we_have_b) = b_ve;
b_ve = b_ve_sparse;
%%
% TODO: check the inputs and report errors if present.
%%
clear b_ve_sparse;

%
% Initial expectations which the inference alrgorithm starts with
%
parameter_mappings_V = DAG_separatedness(gamma_chain_adjacency);
nr_separated = size(parameter_mappings_V, 1);
col_temp = b_ve(:,where_we_have_b(1));
b_ve_temp = col_temp(:,ones(1,network_size));
for(current = 2:nr_separated)
    col_temp = b_ve(:,where_we_have_b(current));
    indeces = find(parameter_mappings_V(current,:));
    b_ve_temp(:,indeces) = col_temp(:,ones(1,size(indeces,2)));
end

% Required: one a_ve parameter for each separate DAG in gamma_chain
% adjacency, per row of V.
% E.g if nr. of separate DAGs is D, a_ve has to be be IxD
a_ve_temp = a_ve;
col_temp = a_ve_temp(:,1);
a_ve = col_temp(:,ones(1,network_size));
for(current = 2:nr_separated)
    col_temp = a_ve_temp(:,current);
    indeces = find(parameter_mappings_V(current,:));
    a_ve(:,indeces) = col_temp(:,ones(1,size(indeces,2)));
end
clear indeces a_ve_temp col_temp;

% shape parameters of the utility nodes are hardcoded
a_ve(:,first_s:end) = repmat(a_utility, [1,size(a_ve,2)-first_s+1]);

% Transposing a_ve i b_ve (it's a convention - it's simply how stuff gets stored for more efficient calculation)
a_ve = a_ve';
b_ve = b_ve';
b_ve_temp = b_ve_temp';
t_init = gamrnd(a_tm, b_tm./a_tm);
v_init = gamrnd(a_ve, b_ve_temp./a_ve);
expectations_markov = zeros(network_size, I, 2);
expectations_markov(:,:,1) = v_init;
clear b_ve_temp where_we_have_b b_ve_temp;
expectations_markov(:,:,2) = expectations_markov(:,:,1);
L_t = t_init;
L_v = expectations_markov(1:last_v,:,1)';
E_t = L_t;
E_v = expectations_markov(1:last_v,:,1)';