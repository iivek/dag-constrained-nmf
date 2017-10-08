% How stuff are organised in this script
% 2 ... size of natural statistics vector
% 3 ... 1st - messages from parents
%       2nd - messages from children from V
%       3rd - messages from children from S
% I ... number of rows in V
messages = zeros(3, I, 2);

% Since the bayes net has the same structure for each row if V,
% message passing can be vectorized for corresponding nodes.

% Messages from parents to us
%
parent_indeces = find( gamma_chain_adjacency(:,current) );
% we have only one parent.
messages(1,:,1) = sum( ...
    expectations_markov(parent_indeces,:,1) .* ...
    (-repmat( a_ve(current,:), [numel(parent_indeces),1] )), 1 );
% second component of messages from parents (the one next to
% the log component of natural statistics vector), for
% corresponding elements accross different bayes nets
messages(1,:,2) = sum( (repmat( a_ve(current,:), [numel(parent_indeces), 1] )-1) ,1);

% % Messages from children to us, children from bayes net
%
child_indeces = find( gamma_chain_adjacency(current,:) );          
if(~isempty(child_indeces)) % should never be
    for(iter = child_indeces)
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