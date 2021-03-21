# nerf-jax
### A JAX rewrite of the NeRF reconstruction technique

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

**This project is still in active development -- for now, try running the tests to get an idea of how it works**

## Setup

1. Install JAX according to [the install guide](https://github.com/google/jax#installation).
2. `python3 -m pip install -r requirements.txt`
3. Download the dataset: `bash download_lego_dataset.sh`

## Training a model

1. First generate an initialization --

```
python generate_sdrf_initialization.py --config config/lego.yml --output experiment/sphere_nerf_penc.pkl
```

2. To visualize intermediate results from the training process, start up Tensorboard:

```
tensorboard --logdir=./logs/sdrf/lego --port 6006
```

3. To start training

```
python train_sdrf.py --config config/lego.yml
```
