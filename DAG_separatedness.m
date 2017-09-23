function [ result ] = DAG_separatedness( adjacency )
    %
    % finds fully separated graphs from DAG
    %
    
    net_size = size(adjacency,1);
    separables = {};
    adjacency_leftover = adjacency;
    last_index = 0;
    while(~isempty(adjacency_leftover))
        last_index = last_index+1
    
        new_roots = 1;
        separable_accumulator = sparse(1,size(adjacency_leftover,1));
        while(true)
            root = new_roots;
            rooted = sparse(1,size(adjacency_leftover,1));
            while(true)
                rooted(1,root) = true;
                [garbage root] = find(adjacency_leftover(root,:));
                if(isempty(root))
                    break
                end
            end
            % we have root + all descendants of root. now we collect and add all
            % parents of rooted (they will repeat and we don't care)
            separable_accumulator = separable_accumulator|rooted;
            separable_accumulator = ~separable_accumulator;
            
            new_roots = any(adjacency_leftover(:,find(rooted))',1)&separable_accumulator;

            separable_accumulator = ~separable_accumulator;
            if(~any(new_roots,2))
                break
            end
        end
    
        unused = find(separable_accumulator==false);
        % remove all used nodes from adjacency matrix
        adjacency_leftover = adjacency_leftover(unused, unused);
        % store what's been found
        separables{last_index} = separable_accumulator;
    
    end
    
    % transforming separables and indexing_compensations to a matrix of
    % separable nodes (will be organised as one separable group per row)
    %
    % cumulative compensations have to be decoupled in order to enable
    % calculation of indices of separables
    result = sparse(size(separables,2), net_size);
    
    propagates_further = (1:net_size);
    for(current=1:size(separables,2))
        result(current, propagates_further(separables{current})) = true;
        propagates_further = propagates_further(find(separables{current} == false));
    end
    
end