% TODO: make the scripts which fetch messages functions
% Stuff is organised as follows:
%
% 2 ... size of natural statistics vector
% 3 ... 1st - messages from parents
%       2nd - messages from children from V
%       3rd - messages from children from S
% I ... number of rows in V
messages = zeros(3, I, 2);

% since the bayes net has the same structure for each row if V,
% message passing can be vectorized for corresponding nodes.

% Messages from parents to us
%
parent_indices = find( gamma_chain_adjacency(:,current) );
if(isempty(parent_indices))
    % we are a topmost parent - a and b from a_ve and b_ve
     messages(1,:,1) = -a_ve(current,:)./b_ve(current,:);
     messages(1,:,2) = (a_ve(current,:)-1);
else
    
    if( isempty(expectations_discrete{current}) )
        % first component of messages from parents, for corresponding
        % elements accross different bayes nets
            messages(1,:,1) = sum( ...
            expectations_markov(parent_indices,:,1) .* ...
            (-a_ve(parent_indices,:)), 1 );
        % second component of messages from parents (the one next to
        % the log component of natural statistics vector), for
        % corresponding elements accross different bayes nets
        messages(1,:,2) = sum( (a_ve(parent_indices,:)-1), 1);
    else
        % first component of messages from parents, for corresponding
        % elements accross different bayes nets
        messages(1,:,1) = sum( ...
            expectations_markov(parent_indices,:,1) .* ...
            (-a_ve(parent_indices,:)) .*...
            expectations_discrete{current}(:,:) ,1 );
        % second component of messages from parents (the one next to
        % the log component of natural statistics vector), for
        % corresponding elements accross different bayes nets
        messages(1,:,2) = sum( (a_ve(parent_indices,:)-1) .*...
            expectations_discrete{current}(:,:) ,1);
    end
    
end

% Messages from children to us, children from bayes net
%
child_indices = find( gamma_chain_adjacency(current,:) );            
if(~isempty(child_indices))
    for(iter = child_indices)
        if( isempty(expectations_discrete{iter}) )
            % destination: messages(2,:,:)
            messages(2,:,1) = messages(2,:,1) + sum( ...
                expectations_markov(iter,:,1) .* ...
                (-a_ve(iter,:)), 1);
            messages(2,:,2) = messages(2,:,2) + sum( a_ve(iter,:), 1);
        else
            %find current's index in expectations_discrete{iter}
            currents_index = find(find( gamma_chain_adjacency(:, iter) )==current);
            messages(2,:,1) = messages(2,:,1) + sum( ...
                expectations_markov(iter,:,1) .* ...
                (-a_ve(iter,:)) .*...
                expectations_discrete{iter}(currents_index,:)...
                ,1);
            messages(2,:,2) = messages(2,:,2) + sum( a_ve(iter,:) .*  ...
                expectations_discrete{iter}(currents_index,:) ,1);        
        end    
    end
end

% Messages from children to us, children from S (hidden layer)
%
% For each vij (element of V), there are nu such children
% if we're here, current is element of V (because current<last_v)
%
% the first entry is sum of expectations of t over nu (over rows)
%
% destination: messages(3,:,:)
%messages(3,:,1) = minus_sum_E_t;
messages(3,:,1) = -masked_sum_over_E_t(:,current)';
messages(3,:,2) = Sig_v(:,current)';