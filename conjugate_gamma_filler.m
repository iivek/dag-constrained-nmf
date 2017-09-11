function [newadj, first_filler] = conjugate_gamma_filler(adjacency)
%conjugate_gamma_filler
%   Returns adjacency matrix that defines a DAG obtained by inserting a node
%   between each parent-child pair. This adjacency matrix is not necessarily
%   topologically sorted, because newly inserted nodes will be appended to
%   the end of the adjacency matrix.

tic
    [rows cols] = size(adjacency);
    % the number of vertices we need to insert is rows-number of vertices
    % that are not parents
    newedges = rows*2-sum(sum(adjacency') == 0);

    newadj = logical(sparse(newedges, newedges));
    % the new vertices will be added at the end of the
    % adjacency matrix that is to be modified (important
    % for keeping track of edge indices)
    
    % Intensive memory usage. To avoid out of memory errors - splitting the
    % copying of matrix into several parts
    %newadj(1:rows, 1:cols) = adjacency;
    
    interval = 10000;
    
    dimension = floor(rows/interval);
    full = dimension;
    leftover = mod(rows,interval);
    if(leftover ~= 0)
        dimension = dimension+1;
    end
    
    intervals = zeros(dimension, 2);
    last = 0;
    for(i = 1:full)        
        last = last+1;
        intervals(i,1) = last;
        last = last + interval-1;
        intervals(i,2) = last;        
    end
    if(leftover ~= 0)
        last = last+1;
        intervals(dimension,1) = last;
        last =last + leftover -1;
        intervals(dimension,2) = last;
    end
    
    for(block = 1:dimension)        
        newadj( intervals(block,1), intervals(block,2) ) = ...
            adjacency( intervals(block,1), intervals(block,2) );                       
    end
    index_of_next = rows+1; % initial position of new vertex
    first_filler = index_of_next;

    for(current_row = 1:rows)
        if(mod(current_row,1000)==0)
            toc
            current_row
            tic
        end
        cols_to_process = find(adjacency(current_row,:));
        if(isempty(cols_to_process))
            continue;
        end
        % disconnect current_row from children
        newadj(current_row, cols_to_process) = zeros(size(cols_to_process));
        % connect current_row to index_of_next
        newadj(current_row, index_of_next) = 1;
        for(iter1 = cols_to_process)        
            % connect index_of_next to ex-children of current_row
            newadj(index_of_next, iter1) = 1;        
        end
        index_of_next = index_of_next+1;
    end
end