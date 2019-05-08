#!/usr/bin/env bash
python main.py --env_id HalfCheetah-v1 --seed=$(echo $RANDOM) --mu=0.8 --episodic
