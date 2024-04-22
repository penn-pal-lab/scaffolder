import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils


class Greedy(nj.Module):

  def __init__(self, wm, act_space, config):

    full_rewfn = lambda s: wm.full_heads['full_reward'](s).mean()[1:]
    if config.train_policy_in_full_wm:
      rewfn = lambda s: wm.full_heads['full_reward'](s).mean()[1:]
    else:
      rewfn = lambda s: wm.heads['reward'](s).mean()[1:]

    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config.loss_scales.critic, config.loss_scales.slowreg, config, 'deter', config.critic, name='critic'),
                 'full_extr': agent.VFunction(full_rewfn, config.full_loss_scales.critic, config.full_loss_scales.slowreg, config, 'full_deter', config.full_critic, name='full_critic')}
    else:
      raise NotImplementedError(config.critic_type)
    self.ac = agent.ImagActorCritic(
        critics, {'extr': 1.0, 'full_extr': 1.0}, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def full_policy(self, latent, state):
    return self.ac.full_policy(latent, state)

  def train(self, imagine_full_policy_in_full_wm, imagine_policy_in_full_wm, imagine_policy_in_wm, start, data):
    return self.ac.train(imagine_full_policy_in_full_wm, imagine_policy_in_full_wm, imagine_policy_in_wm, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}
