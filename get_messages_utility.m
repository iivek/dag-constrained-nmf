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
parent_indices = find( gamma_chain_adjacency(:,current) );
% we have only one parent.
messages(1,:,1) = sum( ...
    expectations_markov(parent_indices,:,1) .* ...
    (-a_ve(parent_indices,:)),1 );
% second component of messages from parents (the one next to
% the log component of natural statistics vector), for
% corresponding elements accross different bayes nets
messages(1,:,2) = sum( (a_ve(parent_indices,:)-1) ,1);

% Messages from children to us, children from bayes net
%
child_indices = find( gamma_chain_adjacency(current,:) );
numchildren = size(child_indices,2);
if(~isempty(child_indices))
    % destination: messages(2,:,:)
    expcs = zeros(numchildren,I);
    cnt = 1;
    for(ci = child_indices)
        if( isempty(expectations_discrete{ci}) )
            expcs(cnt,:) = 1;
        else
            %find current's index in expectations_discrete{iter}
            currents_index = find(find( gamma_chain_adjacency(:, ci) )==current);
            expcs(cnt,:) = expectations_discrete{ci}(currents_index,:);
        end
        cnt = cnt+1;
    end
    messages(2,:,1) =  sum( ...
        expectations_markov(child_indices,:,1) .* ...
        (-a_ve(child_indices,:)) .*expcs ,1);
    messages(2,:,2) = sum( a_ve(child_indices,:) .* expcs ,1);
    
end