import re
import signal
import sys

import embodied
import numpy as np


def train(agent, env, replay, logger, args):
  # SLURM pre-emption logic
  csm = None
  if args.slurm_preempt:
    minutes = 60 * 7 + 45 # every X minutes, requeue this job
    minutes_in_seconds = minutes * 60
    csm = ClusterStateManager(minutes_in_seconds)

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(replay.add)

  # save on slurm preempt
  def slurm_preempt_step(tran, worker):
    if csm is None:
      return
    if csm.should_exit():
      checkpoint.save()
      print("Exiting with exit code", csm.get_exit_code()) # gracefully stop program.
      sys.exit(csm.get_exit_code())
  driver.on_step(slurm_preempt_step)

  # prefill if we are initializing new run (dataset is empty)
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  if checkpoint.exists():
    print('Checkpoint exists, skipping prefill.')
  else:
    print('Prefill train dataset.')
    random_agent = embodied.RandomAgent(env.act_space)
    while len(replay) < max(args.batch_steps, args.train_fill):
      driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0])
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()

class ClusterStateManager:
  def __init__(self, time_to_run):
    self.external_exit = None
    self.timer_exit = False

    signal.signal(signal.SIGTERM, self.signal_handler)
    signal.signal(signal.SIGINT, self.signal_handler)
    signal.signal(signal.SIGALRM, self.timer_handler)
    signal.alarm(time_to_run) # in seconds.

  def signal_handler(self, signal, frame):
    print("Received signal [", signal, "]")
    self.external_exit = signal

  def timer_handler(self, signal, frame):
    print("Received alarm [", signal, "]")
    self.timer_exit = True

  def should_exit(self):
    if self.timer_exit:
      return True

    if self.external_exit is not None:
      return True

    return False

  def get_exit_code(self):
    if self.timer_exit:
      return 3

    if self.external_exit is not None:
      return 0

    return 0
