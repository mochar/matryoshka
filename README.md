Problem: Two datasets with non-matching features and non-matching samples. 
Assume samples of both modalitiies originate from same latent space.
Find common latent space.

Simulated data: MNIST. Every other pixel is modality 1, remaining pixels are modality 2.

Approach: Fit autoencoder on modality 1. Then use one of two models:
- Matryoshka: Freeze weights and extend autoencoder left and right by additional layers that map modality 2 to modality 1, and vice versa.  
- Jian: Replace encoder with a new one that now maps modality 2 to Z, freeze decoder, add additional layer at the end that maps modality 1 to modality 2.

Run:
1. Train initial autoencoder: `python train.py base EPOCHS`
2. Train matryoshka and jian (in two terminals to run parallel): `python train.py {matryoshka,jian} EPOCHS` 
3. Run the notebook `compare.ipynb`

