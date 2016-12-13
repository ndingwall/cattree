# cattree
Native support for categorical variables in sklearn's tree-based models

## Introduction
This module contains four wrappers for scikit-learn's tree-based models that should be considered a proof-of-concept for allowing them to correctly treat categorical variables as single features, without major changes to the API.

Specifically, these models still accept a one-hot encoded numpy array and so can be used in parallel to other scikit-learn models or as part of scikit-learn pipelines.

## Limitations
These classes do not modify the underlying tree-building algorithm and so this implementation is suboptimal! It should be carefully tested before being used in any critical applications (and even then, it probably shouldn't be used anyway). Like I said, proof-of-concept.

## License
This is released under the MIT license. Do what you like.
