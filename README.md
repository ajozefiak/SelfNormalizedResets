# Self-Normalized Resets for Plasticity in Continual Learning

This repository provides the official implementation of Self-Normalized Resets for Plasticity in Continual Learning (ICLR 2025).

## Latest Update
- [June 5, 2025]：An initial code base implementing SNR for the Permuted MNIST experiments is released. Code for convolutional layers is present. The remaining implementations and experiments will be released progressively. 


## Notes and Remarks
- We have released code implementing SNR for both the MLP and CNN architectures.
- We plan to release some technical documentation describing how SNR is precisely implemented, as Algorithm 1 in our paper provides a more general, or abstract, presentation of SNR, omitting implementation details.
- Some artifacts remain from our original code base which implemented SNR and its competitor algorithms using Jax's vectorization in order to allow for massive hyperparameter sweeps to be run in parallel over a single GPU. This will remain and will be refactored as more functionality is released.

## Bibtex Citation
To cite our work, you can use the following:
```
@inproceedings{fariasself,
  title={Self-Normalized Resets for Plasticity in Continual Learning},
  author={Farias, Vivek and Jozefiak, Adam Daniel},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Contact
If you encounter any problem, please file an issue on this GitHub repo.

If you have any question regarding the paper, please contact Adam at [jozefiak@mit.edu](jozefiak@mit.edu).