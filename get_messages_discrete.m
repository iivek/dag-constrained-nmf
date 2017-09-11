% Script for collectiong messages. Data is organised in the following way:
%
% numparents is the size of natural parameters vector
% I
% 2 is the number of messages received
% (matrix organisation different than message passing with Bayes
% net nodes)
messages = zeros(nrparents, I, 2);
messages(:,:,1) = expectations_dirichlet{current};
% messages from child Bayes net node (include expectations from
% their parents)
% reminder: all gammas in the mixture must have the same a para eter,
a_ve_of_parents = a_ve(parents,:);
b_reciprocal_ve_of_parents = a_ve_of_parents.* expectations_markov(parents,:,1);
messages(:,:,2) = (a_ve_of_parents-1).*repmat(expectations_markov(current,:,2), nrparents,1) ...
    - b_reciprocal_ve_of_parents.*repmat(expectations_markov(current,:,1), nrparents,1) ...
    - gammaln(a_ve_of_parents) + a_ve_of_parents.* log(b_reciprocal_ve_of_parents);
