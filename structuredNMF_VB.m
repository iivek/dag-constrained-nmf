function [E_t E_v a_tm b_tm a_ve b_ve M] = structuredNMF_VB(x, a_tm, b_tm, a_ve, b_ve, a_utility, u_dirichlet, adjacency, ...
    EPOCH, start_optimizing_hyperparameters_after, tie_v, tie_a_tm, tie_b_tm, print_period)

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

relevant_indices = 1:network_size;
relevant_indices = relevant_indices(numparents>1);
for(i = relevant_indices)
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
a_ve(:,first_s:end) = a_utility;

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
Sig_t = L_t;
Sig_v = expectations_markov(1:last_v,:,1)';
clear t_init v_init;
temp_log_markov = expectations_markov(1:last_v,:,2);
%
% Parameter initialization ends
%%

relevant_indices = uint32(relevant_indices);
LtLv_raw = zeros(size(x_raw));

calculate_bound = true;
boundOld = -Inf;
tictocs = zeros(1,EPOCH);
mean_train = sum(sum(x_raw))./full(sum(sum(M)));
gammalnX_raw = gammaln(double(x_raw) + 1);
start_from = 1;
bounds = [];
for e=start_from:EPOCH,
    tic;
    
    %LtLv = L_t*L_v; % this can potentially be a large matrix - but we only
    %need entries on same locations where x is observed
    LtLv_raw = sum(L_t(rows_mask,:).*L_v(:,cols_mask)',2);
    tmp = sparse(W,K);
    tmp(linear_mask) = x_raw./LtLv_raw;
    Sig_t = L_t.*(tmp*L_v');
    Sig_v = L_v.*(L_t'*tmp);
    
    % Updating the expectations of T
    alpha_tm = a_tm + Sig_t;
    beta_tm = 1./( a_tm./b_tm + M*E_v' );
    E_t = alpha_tm.*beta_tm;
    masked_sum_over_E_t = E_t'*M;
    
    % Updating the expectations of V
    for(current = 1:last_v) 
        get_messages_V; % "messages" are info coming from the Markov blanket of the currently processed node in the graphical model. The so called "message passing" framework
        natural_parameters = sum(messages);
        a = natural_parameters(:,:,2)+1;
        b = -1./natural_parameters(:,:,1);
        % having fully obtained the mesages, updated natural parameter
        % vector is used to reestimate sufficient statistics
        expectations_markov(current,:,1) = a.*b;
        temp_log_markov(current,:) = psi(a)+log(b);
        %expectations_markov(current,:,2) = psi(a)+log(b);
    end
    E_v = expectations_markov(1:last_v,:,1)';
    
    % Now it's a good moment to efficielntly calculate bound. Note that Lt
    % and Lv and expectations_markov have not yet been refreshed
    if rem(e, 2)==0, fprintf(1, '*'); end;
    calculate_bound = rem(e, print_period)==0 | e==EPOCH;
    if calculate_bound,
        bound_procedure;
        
        diff = bound-boundOld;
        boundOld = bound;
        fprintf(1, '\nBound = %f\tdiff = %f\tlast_elapsed = %f\n', boundOld, diff, tictocs(e));
        rrse_train = sse(x_raw - EtEv_raw)./sse(x_raw - ones(size(x_raw,1),1)*mean_train );
        fprintf(1, '\trrse_train = %f\n', rrse_train);
        bounds = [bounds, bound];

        if(boundOld > bound)
            warning('Convergence violated')
        end
    end
       
    expectations_markov(1:last_v,:,2) = temp_log_markov;
    
    % Updating the utility, inverse-gamma nodes, which have been added to
    % make the gamma chains conjugate.
    for(current = first_s:network_size)
        get_messages_utility;
        natural_parameters = sum(messages);
        a = natural_parameters(:,:,2)+1;
        b = -1./natural_parameters(:,:,1);        
        % having fully obtained the mesages, the updated natural parameter
        % vector is now used to reestimate sufficient statistics
        expectations_markov(current,:,1) = a.*b;
        expectations_markov(current,:,2) = psi(a)+log(b); 
    end
    
    % Updating the mixture selection models, alongside with their Dirichlet
    % priors. Mixture models is added to gamma nodes which have multiple
    % parents, according to the specified adjacency matrix.
    numparents = sum(gamma_chain_adjacency);
    relevant_indices = find(numparents > 1);
    for(current = relevant_indices)
        % A safe way to get the current's parents:
        parents = find( gamma_chain_adjacency(:, current) );
        nrparents = size(parents,1); 
        % Dirichlet distributions
        %
        % we get a Dirichlet distribution with following parameters u:
        u_parameters = expectations_discrete{current}( : ,:) + ...
            u_dirichlet{current};
        % ...and the expectation of such a Dirichlet is
        expectations_dirichlet{current} = psi(u_parameters) - ...
            ones(size(u_parameters))*diag( psi(sum(u_parameters,1)) );
        % Discrete distributions:
        get_messages_discrete;
        natural_parameters = sum(messages,3);
        %
        % Updating expectations:
        % They are different from other exponential family distributions in
        % the way log-partition function  gets determined; here we require
        % natural parameters to form discrete likelihoods
        expectations_discrete{current} = exp(natural_parameters)./repmat(Z,size(natural_parameters,1),1);
    end
    
    L_t = exp(psi(alpha_tm)).*beta_tm;
    L_v = exp(expectations_markov(1:last_v,:,2)');
    
    if(e>start_optimizing_hyperparameters_after)
        %% Hyperparameter optimization
        %.
        switch tie_b_tm,
            case 'free',
                b_tm = E_t;
            case 'rows',
                b_tm = repmat(sum(a_tm.*E_t, 1)./sum(a_tm,1), [W 1]);
            case 'cols',
                b_tm = repmat(sum(a_tm.*E_t, 2)./sum(a_tm,2), [1 I]);
            case 'tie_all',
                b_tm = sum(sum(a_tm.*E_t))./sum(a_tm(:)).*ones(W, I);
            % case 'clamp', do nothing
        end;
        
        % Newton-Rapshon numerical optimization for shape parameters
        if ~strcmp( tie_a_tm, 'clamp'),
            Z = E_t./b_tm - (log(L_t) - log(b_tm));
            switch tie_a_tm,
                case 'free',
                    a_tm = gnmf_solvebynewton(Z, a_tm);
                case 'rows',
                    a_tm = gnmf_solvebynewton(sum(Z,1)/W, a_tm);
                case 'cols',
                    a_tm = gnmf_solvebynewton(sum(Z,2)/I, a_tm);
                case 'tie_all',
                    a_tm = gnmf_solvebynewton(sum(Z(:))/(W*I), a_tm);
                % case 'clamp', do nothing
            end;
        end;
        
        % Updating the shapes and scales related to factor V
        % For a_ve, b_ve optimization has to be done by traversing the DAGs.
        % This code makes 2 options available:
        %   1. clamped a_ve, clamped b_ve -> 'clamp'
        %   2. rowwise tied a_ve (as dictated by parameter_mappings_V), free b_ve -> 'free'
        %   3. clamped a_ve, free b_ve -> 'free'
        switch(tie_v)
            case 'free',
                % TODO: more extensive testing - are 10 iterations of
                % Newton-Raphson enough?
                nonzero_bs = (b_ve(:,1) ~= 0)';   % supposing graphical models have the same structure for each row of V
                for( iter = 1:size(parameter_mappings_V,1) )
                    currentDAG = parameter_mappings_V(iter,:);
                    mask = currentDAG & ~nonzero_bs;
                    
                    indeces = find(mask);
                    Z=0;
                    for(i = indeces)
                        parent_indices = find( gamma_chain_adjacency(:,i) );
                        if( isempty(expectations_discrete{i}) )
                            Z = Z + ...
                                expectations_markov(i,:,1).*sum(expectations_markov(parent_indices,:,1) ,1) ...
                                -expectations_markov(i,:,2) ...
                                -sum(expectations_markov(parent_indices,:,2) ,1);
                        else
                            Z = Z + ...
                                expectations_markov(i,:,1).*sum(expectations_markov(parent_indices,:,1).*expectations_discrete{i}(:,:) ,1) ...
                                -expectations_markov(i,:,2) ...
                                -sum(expectations_markov(parent_indices,:,2).*expectations_discrete{i}(:,:) ,1);
                        end
                    end
                    % accumulating from non-topmost nodes done
                    mask = currentDAG & nonzero_bs;
                    indeces = find(mask);               
                    % b_ve (topmost only)
                    b_ve(indeces,:) = expectations_markov(indeces,:,1);
                    % topmost done.
                    Z = Z + sum( expectations_markov(indeces,:,1)./b_ve(indeces,:) ...
                        -expectations_markov(indeces,:,2) + ...
                        log(b_ve(indeces,:)), 1);
                    
                    Z = Z/sum(parameter_mappings_V(iter,:),2); % to get the form prepared for solving
                    indeces = find(currentDAG);
                    a_ve(indeces,:) = gnmf_solvebynewton(Z, a_ve(indeces,:));
                end
            case 'semifree',
                nonzero_bs = (b_ve(:,1) ~= 0)';   % supposing graphical models have the same structure for each row of V
                for( iter = 1:size(parameter_mappings_V,1) )
                    currentDAG = parameter_mappings_V(iter,:);
                                                            
                    mask = currentDAG & nonzero_bs;     % TODO: don't calculate this at each iteration - store it
                    indeces = find(mask);      
                    % b_ve (topmost only)
                    b_ve(indeces,:) = expectations_markov(indeces,:,1);
                    
                end
                % free parameters end
            case 'clamp'
                % do nothing                
        end
        
    end
    
    tictocs(1,e) = toc;
    
end
