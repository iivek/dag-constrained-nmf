% In this version of free energy calculation we don't require
% exp(psi(alpha_tm)).*beta_tm == L_t, but only sig_t and sig_v

EtEv_raw = sum(E_t(rows_mask,:).*E_v(:,cols_mask)',2);
bla_raw = sum(L_t(rows_mask,:).*log(L_t(rows_mask,:)).*L_v(:,cols_mask)',2);

bound = -sum(sum(EtEv_raw +  gammalnX_raw )) ...
    + sum(sum( -x_raw.*( bla_raw./LtLv_raw -  log(LtLv_raw) )  )) ...
    + sum(sum(-a_tm./b_tm.*E_t - gammaln(a_tm) + a_tm.*log(a_tm./b_tm)  )) ...
    + sum(sum( gammaln(alpha_tm) + alpha_tm.*(log(beta_tm) + 1)  ));

part1 = 0;
part2 = 0;
for(current = 1:last_v)
    % note that, in case there is no structure over V, a and b become
    % columns of alpha_ve and beta_ve from Cemgil's paper

    % contribution to bound - expectation of probability part
    %
    parent_indices = find( gamma_chain_adjacency(:,current) );
    numparents = size(parent_indices,1);
    
    if(numparents == 0)
        % we're dealing with free a and b params.
        shape = a_ve(current,:);
        scale = b_ve(current,:)./a_ve(current,:);
        part1 = part1 + sum( ...
            (shape-1).*expectations_markov(current,:,2) ...
            - 1./scale.*expectations_markov(current,:,1) - gammaln(shape)...
            - shape.*log(scale) );
    else if(numparents == 1)
        shapes =  repmat(a_ve(current,:),[numparents,1]);
        scales = 1./(shapes.*expectations_markov(parent_indices,:,1));
        part1 = part1 + sum( sum( ( ...
            (shapes - 1).*repmat(expectations_markov(current,:,2), numparents,1) - ...
            1./scales.*repmat(expectations_markov(current,:,1), numparents,1 ) - ...
            gammaln( shapes ) - ...
            shapes .* log( scales ) ) ));
        else
            shapes =  repmat(a_ve(current,:),[numparents,1]);
            scales = 1./(shapes.*expectations_markov(parent_indices,:,1));
            part1 = part1 + sum( sum( expectations_discrete{current}(:,:) .* ( ...
               (shapes - 1).*repmat(expectations_markov(current,:,2), numparents,1) - ...
                1./scales.*repmat(expectations_markov(current,:,1), numparents,1 ) - ...
                gammaln( shapes ) - ...
                shapes .* log( scales )   ) ));
        end
    end
    
    get_messages_V;
    
    natural_parameters = sum(messages);
    a = natural_parameters(:,:,2)+1;
    b = -1./natural_parameters(:,:,1);
    
    % note that, in case there is no structure over V, a and b become
    % columns of alpha_ve and beta_ve from Cemgil's paper
    
    % contribution to bound - the entropy part
    % - entropy of gamma pdf having natural parameters as calculated
    % above
    %
    psi_a_ = expectations_markov(current,:,2)-log(b);
    part2 = part2 + sum(-(a-1).*psi_a_+log(b)+ a + gammaln(a));
end

bound = bound + part1 + part2;

%
% The utility elements from the bayes network
%
part1 = 0;
part2 = 0;
for(current = first_s:network_size)
    % Note: if we're here, we never have more than one parent, so
    % stuff below can be simplified
    %
    % contribution to bound - expectation of probability part
    %
    parent = find( gamma_chain_adjacency(:,current) );
    %numparents = size(parent_indices,1);    % will always be equal to one
    shape = a_ve(current,:);
    scale = 1./(shape.*expectations_markov(parent,:,1));
    part1 = part1 + sum( ...
        (shape-1).*expectations_markov(current,:,2) ...
        - 1./scale.*expectations_markov(current,:,1) - gammaln(shape)...
        - shape.*log(scale) );
    
%     % contribution to bound - the entropy part
%     get_messages_utility;
%     
%     natural_parameters = sum(messages);
%     a = natural_parameters(:,:,2)+1;
%     b = -1./natural_parameters(:,:,1);
    
%     psi_a_ = psi(a);
%     part2 = part2 + sum(-(a-1).*psi_a_+log(b) + a + gammaln(a));
    % sum the bound part components across Bayes networks and
    % add them to bound
end

bound = bound + part1 + part2;
bound = bound+entropies;
%
% Mixture selection part of Bayes net - discrete distros together with
% Dirichlet priors
%
numparents = sum(gamma_chain_adjacency);
relevant_indices = find(numparents > 1);
for(current = relevant_indices)
    % a safe way to get the current's parents:
    parents = find( gamma_chain_adjacency(:, current) );
    nrparents = size(parents,1);
    
    %Discrete pdfs:
    %
    get_messages_discrete;
    
    % contribution to the bound - expectation of log probability:
    Z_divided_by_normalization_constants = ...
        repmat(sum(expectations_dirichlet{current}),nrparents,1)./expectations_dirichlet{current};
    probabilities = ...
        1./Z_divided_by_normalization_constants;
    %sum(probabilities) == 1
    bound_part = sum(probabilities .* expectations_discrete{current}(: ,:),1);
    
    % contribution to the bound -entropy of variational factor:
    regular_entropies_indices = find(...
        sum(expectations_discrete{current}(:,:) ==0) ==0);
    %
    bound_part = sum(bound_part) - ...
        sum(sum(expectations_discrete{current}(:,regular_entropies_indices).* ...
        log(expectations_discrete{current}(:,regular_entropies_indices)),1));        % entropy
    bound = bound + bound_part;
    
    % Dirichlet pdfs
    %
    % we get a Dirichlet distribution with following parameters u:
    u_parameters = expectations_discrete{current}( : ,:) + ...
        u_dirichlet{current};
    %
    % contribution to bound - expectation of log probability:
    bound_part = sum((u_dirichlet{current}-1).*expectations_dirichlet{current}) ...
        + gammaln(sum(u_dirichlet{current})) ...
        - sum(gammaln(u_dirichlet{current}));
    
    % contribution to bound - entropy of variational factor:
    sum_u = sum(u_parameters);
    bound_part = bound_part ...
        + sum(gammaln(u_parameters))-gammaln(sum_u) ...
        +(sum_u-nrparents).*psi(sum_u)- ...
        sum((u_parameters-1).*psi(u_parameters));
    
    bound = bound + sum(bound_part,2);
    if(isnan(bound))
        % Just a precaution, but should not happen
        warning('Broken dirichscrete')
    end
end
