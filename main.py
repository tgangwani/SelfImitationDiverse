import os
import sys
import argparse
import gym

from imitation.algo import ppo
from imitation.network.discriminator import Discriminator
from imitation.network.mlp_policy import MlpPolicy
from imitation.common import tf_util as U
from imitation.common.misc_util import set_global_seeds
from imitation.common.monitor import Monitor

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of XXX")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)

    # Network Configuration
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--discriminator_hidden_size', type=int, default=100)

    # Training Configuration
    parser.add_argument('--policy_entcoef', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--vf_coef', help='coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--discriminator_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)

    # Self-imitation Configuration
    parser.add_argument('--mu', type=float, default=1.)
    parser.add_argument('--pq_replay_size', help='Entries in priority queue (# trajectories)', type=int, default=10)
    parser.add_argument('--episodic', help='provide reward only at the last timestep', dest='episodic', action='store_true')

    return parser.parse_args()

def main(args):
    U.make_session(num_cpu=args.num_cpu).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    if str(env.__class__.__name__).find('TimeLimit') >= 0:
        from imitation.common.env_wrappers import TimeLimitMaskWrapper
        env = TimeLimitMaskWrapper(env)

    env = Monitor(env, filename=None, allow_early_resets=True)
    env.seed(args.seed)
    env_name = args.env_id.split("-")[0]
    data_dump_path = os.path.join(os.getcwd(), 'Results', env_name, '_'.join(['log', str(args.seed)]))

    if args.episodic:
        from imitation.common.env_wrappers import EpisodicEnvWrapper
        env = EpisodicEnvWrapper(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=policy_fn.hid_size, num_hid_layers=2)

    policy_fn.hid_size = args.policy_hidden_size
    discriminator = Discriminator(env, args.discriminator_hidden_size, args.discriminator_entcoeff)

    ppo.learn(env, policy_fn, discriminator,
            d_step=args.d_step,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoef=args.policy_entcoef, vf_coef=args.vf_coef,
            optim_epochs=5, optim_stepsize=1e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, d_stepsize=1e-4,
            max_timesteps=args.num_timesteps,
            schedule='constant',
            data_dump_path=data_dump_path,
            mu=args.mu,
            pq_replay_size=args.pq_replay_size)

    env.close()

if __name__ == '__main__':
    print(sys.argv)
    args = argsparser()
    main(args)
