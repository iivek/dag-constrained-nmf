## Synopsis

This Nonnegative Matrix Factorization formulation was motivated by collaborative filtering applications where taxonomy of items is readily available.
It features graph-based (DAG) constraints over latent factors. More precisely, Markov chains are defined over latent factors according to the given DAG structure. The model can handle nodes with multiple parents.

## Motivation

Consider movie recommendation problem where taxonomy of movie genres is known. For example, suppose that some horror fan liked a slasher horror movie. Then it is reasonable to assume that he/she will also like other slashers. On the other hand, suppose that he/she did not like a zombie horror he/she watched - then it is a good guess that he/she will not enjoy other zombie movies.
These assumptions will be modeled as two Markov chains, the first one enforcing correlations between slasher movie ratings and the second one describing correlations between zombie movie ratings.

Collaborative filtering methods collect preferences from other users and their recommendation is based on community-driven tastes of similar users. In case rating or preference data is scarce or the community that provides ratings is rather small, collaborative filtering methods will run into issues. What may help is introducing additional domain-specific knowledge in the model.
This model includes extra information in form of item taxonomies and hope is that it will lead to better-informed recommendations in cases where number of ratings per item is small.

## Example

Decomposition results on a small synthetic dataset are shown below.
![Reconstruction](/examples/example-reconstruction.png)
![Latent factors](/examples/example-factors.png)
The generated (oracle) matrix is low-rank, with known correlations in the latent factors. The algorithm has been presented with only 10% of the oracle data with the goal of reconstructing the rest. In case correlations are embedded into the algorithm, reconstruction is closer to the oracle as compared to the correlation-ignorant decomposition.

## About the Model

The presented probabilistic model is conjugate. Its parameters are iteratively calculated using variational Bayesian learning; noise is Poissonian and matrix elements are Gamma distributed. Markov chains are built over Gamma variables belonging to the right factor matrix; in case a variable has multiple parents, a discrete mixture model with Dirichlet priors is introduced there (basically a child can be correlated to a single parent only and this selection is probabilistically modeled).
The core of the generative model, Poisson-Gamma NMF model, as well as its inference is the same as in [1]; novelty are the additional hierarchical constraints over right matrix.

## Interpretation

Let ***X*** (*n* x *m*) be matrix of ratings, where each row contains ratings related to a user (n users, m movies). Most commonly ***X*** has many unobserved values, as the only observed data are the user ratings. The algorithm finds low-rank factorizations of ***X***, ***X***=***T***&ast;***V***, where ***T*** (*n* x *i*) is activation matrix and ***V*** (*i* x *m*) matrix of latent factors, organised rowwise. Rating prediction then boils down to filling out the missing values of ***X***, which is done by multiplication of the two factor matrices.
Note that DAG constraints are imposed on each of the rows of ***V***.

## References

[1] A. T. Cemgil, “Variational Bayesian nonnegative matrix factorization.” [Online]. Available: http://www.cmpe.boun.edu.tr/~cemgil/bnmf/. [Accessed: 01-Jul-2013].

If you have used this code, please cite this repo as:
I.Ivek, "DAG-constrained Variational Bayesian NMF." [Online]. Available: https://github.com/iivek/dag-constrained-nmf

## License

Published under GPL-3.0 License.
