import time
from collections import deque
import numpy as np
import tensorflow as tf

from imitation.priority_buffer import TrajReplay
from imitation.common.adam import Adam
from imitation.common import logger, dataset, tf_util as U
from imitation.common.math_util import explained_variance
from imitation.common.misc_util import zipsame, fmt_row, OrderedDefaultDict

np.set_printoptions(precision=3)

def rollout_generator(pi, env, discriminator, timesteps_per_batch, running_paths, stochastic):
    t = 0
    running_path_idx = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    new_time_limit = True  # marks if the previous episode ended due to enforced time-limits
    rew_ER = 0.0    # environmental rewards
    ob = env.reset()

    cur_ep_len = 0  # length of current episode
    cur_ep_ret_ER = 0   # environmental returns in current episode
    ep_lens = []  # lengths of episodes completed in this rollout
    ep_rets_oracle = []  # episodic oracle returns (only used for plotting!)

    # Initialize history arrays
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    rews_ER = np.zeros(timesteps_per_batch, 'float32')
    vpreds = np.zeros(timesteps_per_batch, 'float32')
    vpreds_ER = np.zeros(timesteps_per_batch, 'float32')
    news = np.zeros(timesteps_per_batch, 'int32')
    news_time_limit = np.zeros(timesteps_per_batch, 'int32')
    acs = np.array([ac for _ in range(timesteps_per_batch)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, vpred_ER = pi.act(stochastic, ob)

        running_paths[running_path_idx]['obs'].append(ob)
        running_paths[running_path_idx]['acs'].append(ac)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % timesteps_per_batch == 0:

            # Obtain discriminator rewards
            rews = np.squeeze(discriminator.get_reward(obs, acs), axis=1)

            yield {"obs" : obs, "rews" : rews, "rews_ER" : rews_ER, "vpreds" : vpreds, "vpreds_ER" : vpreds_ER, "news" : news,
                    "news_time_limit" : np.append(news_time_limit, new_time_limit), "acs" : acs, "prevacs" : prevacs, "nextvpred": vpred * (1 - new),
                    "nextvpred_ER": vpred_ER * (1 - new), "ep_lens" : ep_lens, "ep_rets_oracle": ep_rets_oracle}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets_oracle = []
            ep_lens = []

        i = t % timesteps_per_batch
        obs[i] = ob
        vpreds[i] = vpred
        vpreds_ER[i] = vpred_ER
        news[i] = new
        news_time_limit[i] = new_time_limit
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew_ER, new, info = env.step(ac)
        new_time_limit = 1 if 'timestep_limit_reached' in info.keys() else 0
        rews_ER[i] = rew_ER
        cur_ep_ret_ER += rew_ER
        cur_ep_len += 1

        if new:
            running_paths[running_path_idx]['return'] = cur_ep_ret_ER
            running_path_idx += 1
            ep_rets_oracle.append(info['episode']['r'])  # Retrieve the oracle score inserted by monitor.py
            ep_lens.append(cur_ep_len)
            cur_ep_ret_ER = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(rollout, gamma, lam):
    """
    Calculate, independently for discriminator and environmental rewards, the advantage
    using GAE, and critic-target using TD (lambda)
    """
    new = np.append(rollout["news"], 0)
    new_time_limit = rollout["news_time_limit"]
    vpred = np.append(rollout["vpreds"], rollout["nextvpred"])
    vpred_ER = np.append(rollout["vpreds_ER"], rollout["nextvpred_ER"])
    rew = rollout["rews"]
    rew_ER = rollout["rews_ER"]
    T = len(rew)
    rollout["adv"] = gaelam = np.empty(T, 'float32')
    rollout["adv_ER"] = gaelam_ER = np.empty(T, 'float32')
    lastgaelam = 0; lastgaelam_ER = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        delta_ER = rew_ER[t] + gamma * vpred_ER[t+1] * nonterminal - vpred_ER[t]
        gaelam[t] = lastgaelam = (delta + gamma * lam * nonterminal * lastgaelam) * (1-new_time_limit[t+1])
        gaelam_ER[t] = lastgaelam_ER = (delta_ER + gamma * lam * nonterminal * lastgaelam_ER) * (1-new_time_limit[t+1])
    rollout["tdlamret"] = rollout["adv"] + rollout["vpreds"]
    rollout["tdlamret_ER"] = rollout["adv_ER"] + rollout["vpreds_ER"]

def learn(env, policy_fn, discriminator, *,
        d_step, timesteps_per_batch,
        clip_param, entcoef, vf_coef,
        optim_epochs, optim_stepsize, optim_batchsize,
        gamma, lam, d_stepsize,
        max_timesteps, schedule, # annealing for stepsize parameters (epsilon and adam)
        data_dump_path, mu, pq_replay_size):

    log_record = U.logRecord(data_dump_path)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # TD (lambda) return with discriminator rewards
    ret_ER = tf.placeholder(dtype=tf.float32, shape=[None]) # TD (lambda) return with environmental rewards
    oldvpred = tf.placeholder(dtype=tf.float32, shape=[None])
    oldvpred_ER = tf.placeholder(dtype=tf.float32, shape=[None])

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    ent = pi.pd.entropy()
    meanent = U.mean(ent)
    pol_entpen = (-entcoef) * meanent

    logratio = tf.clip_by_value(pi.pd.logp(ac) - oldpi.pd.logp(ac), -20., 20.)  # clip for numerical stability
    ratio = tf.exp(logratio) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration

    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    # Clipped Critic losses
    vpredclipped = oldvpred + tf.clip_by_value(pi.vpred - oldvpred, -tf.abs(oldvpred)*clip_param, tf.abs(oldvpred)*clip_param)
    vf_losses1 = tf.square(pi.vpred - ret)
    vf_losses2 = tf.square(vpredclipped - ret)
    vf_loss = vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    vpredclipped = oldvpred_ER + tf.clip_by_value(pi.vpred_ER - oldvpred_ER, -tf.abs(oldvpred_ER)*clip_param, tf.abs(oldvpred_ER)*clip_param)
    vf_losses1 = tf.square(pi.vpred_ER - ret_ER)
    vf_losses2 = tf.square(vpredclipped - ret_ER)
    vf_ER_loss = vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    total_pi_loss = pol_surr + pol_entpen
    total_vf_loss = vf_ER_loss + vf_loss
    pi_losses = [pol_surr, pol_entpen, meanent]
    vf_losses = [vf_loss, vf_ER_loss]
    pi_loss_names = ["pol_surr", "pol_entpen", "ent"]
    vf_loss_names = ["vf_loss", "vf_ER_loss"]
    loss_names = pi_loss_names + vf_loss_names

    # Separate actor-critic paramters into 2 lists
    var_list = pi.get_trainable_variables()
    pi_var_list = [v for v in var_list if v.name.split("/")[1].startswith("pol")]
    pi_var_list += [v for v in var_list if v.name.split("/")[1].startswith("logstd")]
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith("vff")]
    assert not (set(var_list) - set(pi_var_list) - set(vf_var_list)), "Missed variables."

    pi_lossandgrad = U.function([ob, ac, atarg, lrmult], pi_losses + [U.flatgrad(total_pi_loss, pi_var_list)])
    vf_lossandgrad = U.function([ob, ret, ret_ER, oldvpred, oldvpred_ER, lrmult], vf_losses + [U.flatgrad(total_vf_loss, vf_var_list)])

    # Optimizers for actor, critic and discriminator
    pi_adam = Adam(pi_var_list, epsilon=1e-5)
    vf_adam = Adam(vf_var_list, epsilon=1e-5)
    d_adam = Adam(discriminator.get_trainable_variables(), epsilon=1e-5)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_pi_losses = U.function([ob, ac, atarg, lrmult], pi_losses)
    compute_vf_losses = U.function([ob, ret, ret_ER, oldvpred, oldvpred_ER, lrmult], vf_losses)

    # Initiate priority-queue replay
    pq_replay = TrajReplay(capacity=pq_replay_size)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    running_paths = OrderedDefaultDict()  # dictionary to save competed paths (paths == episodes == trajectories)
    rg = rollout_generator(pi, env, discriminator, timesteps_per_batch, running_paths, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rets_oracle_buffer = deque(maxlen=100)   # rolling buffer for episodic environmental returns

    # == Main loop ==
    while timesteps_so_far < max_timesteps:

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("[%s] ********** Iteration %i ************" % (time.strftime('%x %X %z'), iters_so_far))

        # ------------------ Update G (policy) ------------------
        logger.log("Optimizing Policy...")

        # Rollout using current policy
        rollout = rg.__next__()
        add_vtarg_and_adv(rollout, gamma, lam)

        # Add completed episodes to priority replay, and sync!
        for path in list(running_paths.values())[:-1]:
            pq_replay.add_path(path)
        pq_replay.sync()

        # Reclaim memory
        for k in list(running_paths.keys())[:-1]:
            del running_paths[k]

        pol_obs, pol_acs, atarg, atarg_ER, tdlamret, tdlamret_ER = rollout["obs"], rollout["acs"], \
                rollout["adv"], rollout["adv_ER"], rollout["tdlamret"], rollout["tdlamret_ER"]

        # Standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        atarg_ER = (atarg_ER - atarg_ER.mean()) / atarg_ER.std()

        d = dataset.Dataset(dict(obs=pol_obs, acs=pol_acs, atarg=atarg, atarg_ER=atarg_ER, vtarg=tdlamret,
            vtarg_ER=tdlamret_ER, oldvpred=rollout["vpreds"], oldvpred_ER=rollout["vpreds_ER"]), deterministic=pi.recurrent)

        # update running mean/std for policy
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(pol_obs)

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        # PPO optimization
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):

                # "g" is policy gradient using discriminator rewards, while
                # "g_ER" is policy gradient using environmental rewards
                *pi_newlosses, g = pi_lossandgrad(batch["obs"], batch["acs"], batch["atarg"], cur_lrmult)
                *pi_newlosses_ER, g_ER = pi_lossandgrad(batch["obs"], batch["acs"], batch["atarg_ER"], cur_lrmult)
                pi_newlosses_avg = np.average(np.array([pi_newlosses, pi_newlosses_ER]), axis=0, weights=[mu, 1-mu])

                # Update policy with averaged gradient
                g_avg = np.average(np.array([g, g_ER]), axis=0, weights=[mu, 1-mu])
                pi_adam.update(g_avg, optim_stepsize * cur_lrmult)

                # Update both critics
                *vf_newlosses, g_vf = vf_lossandgrad(batch["obs"], batch["vtarg"], batch["vtarg_ER"], batch["oldvpred"], batch["oldvpred_ER"], cur_lrmult)
                vf_adam.update(g_vf, optim_stepsize * cur_lrmult)

                losses.append(list(pi_newlosses_avg) + vf_newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            pi_newlosses = compute_pi_losses(batch["obs"], batch["acs"], batch["atarg"], cur_lrmult)
            pi_newlosses_ER = compute_pi_losses(batch["obs"], batch["acs"], batch["atarg_ER"], cur_lrmult)
            pi_newlosses_avg = np.average(np.array([pi_newlosses, pi_newlosses_ER]), axis=0, weights=[mu, 1-mu])
            vf_newlosses = compute_vf_losses(batch["obs"], batch["vtarg"], batch["vtarg_ER"], batch["oldvpred"], batch["oldvpred_ER"], cur_lrmult)
            losses.append(list(pi_newlosses_avg) + vf_newlosses)

        meanlosses = np.mean(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)

        logger.record_tabular("ev_tdlam_before", explained_variance(rollout["vpreds"], tdlamret))
        logger.record_tabular("ev_tdlam_before_ER", explained_variance(rollout["vpreds_ER"], tdlamret_ER))

        lens, rets_oracle = rollout["ep_lens"], rollout["ep_rets_oracle"]
        log_record.insert(key='environment_episodic_returns', value=rets_oracle)
        rets_oracle_buffer.extend(rets_oracle)
        lenbuffer.extend(lens)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpOracleRetMean", np.mean(rets_oracle_buffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += timesteps_per_batch

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.dump_tabular()

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_names))

        d_losses = [] # list of tuples, each of which gives the loss for a d_step
        # d_step updates for the discriminator
        for pol_data, expert_data in zip(
                dataset.iterbatches((pol_obs, pol_acs), batch_size=(timesteps_per_batch // d_step)),
                dataset.iterbatches((pq_replay.obs, pq_replay.acs, pq_replay.wts), batch_size=(len(pq_replay) // d_step))):

            pol_obs_mb, pol_acs_mb = pol_data
            expert_obs_mb, expert_acs_mb, expert_wts_mb = expert_data
            expert_wts_mb /= np.sum(expert_wts_mb)

            # update input (ob+ac) mean/std for discriminator
            if hasattr(discriminator, "in_rms"): discriminator.in_rms.update(
                    np.concatenate((
                        np.concatenate((pol_obs_mb, expert_obs_mb), axis=0),
                        np.concatenate((pol_acs_mb, expert_acs_mb), axis=0)),
                        axis=1))

            *newlosses, g = discriminator.lossandgrad(pol_obs_mb, pol_acs_mb, expert_obs_mb, expert_acs_mb, expert_wts_mb)
            d_adam.update(g, d_stepsize * cur_lrmult)
            d_losses.append(newlosses)

        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        iters_so_far += 1

    log_record.dump()
