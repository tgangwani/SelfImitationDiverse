This repo contains code for our paper [Learning Self-Imitating Diverse Policies](https://arxiv.org/abs/1805.10309) published at ICLR 2019. 

The code was tested with the following packages:

* python 3.5.2
* tensorflow 1.4.0
* gym  0.9.2

## Running command
To train a self-imitation agent in an episodic reward environment, use:

```
python main.py --env_id HalfCheetah-v1 --seed=$(echo $RANDOM) --mu=0.8 --episodic
```

The parameter 'mu' is as defined in the paper (Equation 5.)

## Coming soon..
Code using SVPG for diverse multi-agent training

## Acknowledgements
The code is built on, and uses many utils from [OpenAI baselines](https://github.com/openai/baselines)
