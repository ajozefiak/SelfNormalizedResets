# Self-Normalized Resets for Plasticity in Continual Learning

<p align="center">
<a href="https://colab.research.google.com/drive/18jHVgPTH4CM9hlvnMvapbd4CMgi1x_2Q?usp=share_link" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

This repository provides the official implementation of Self-Normalized Resets for Plasticity in Continual Learning (ICLR 2025).

## Latest Update
- [August 6, 2025]: The all_shakespeare.txt dataset and a temporary barebones implementation of the Permuted Shakespeare experiment have been released. 
- [June 5, 2025]：An initial code base implementing SNR for the Permuted MNIST experiments is released. Code for convolutional layers is present. The remaining implementations and experiments will be released progressively. 


## Notes and Remarks
- We have released code implementing SNR for both the MLP and CNN architectures.
- We plan to release some technical documentation describing how SNR is precisely implemented, as Algorithm 1 in our paper provides a more general, or abstract, presentation of SNR, omitting implementation details.
- Some artifacts remain from our original code base which implemented SNR and its competitor algorithms using Jax's vectorization in order to allow for massive hyperparameter sweeps to be run in parallel over a single GPU. This will remain and will be refactored as more functionality is released.

## Example Usage (Running on Google Colab)
Check out the Google Colab [here](https://colab.research.google.com/drive/18jHVgPTH4CM9hlvnMvapbd4CMgi1x_2Q?usp=share_link)!

<a href="https://colab.research.google.com/drive/18jHVgPTH4CM9hlvnMvapbd4CMgi1x_2Q?usp=share_link" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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