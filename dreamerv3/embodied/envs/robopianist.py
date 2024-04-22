import functools
import os
from pathlib import Path

import embodied
import numpy as np

class RoboPianist(embodied.Env):
  def __init__(self, task, render_image=False, record=False, **kwargs):
    import robopianist
    from robopianist import suite
    import robopianist.wrappers as robopianist_wrappers
    import dm_env
    import dm_env_wrappers
    
    env = suite.load(
      environment_name=task,
      task_kwargs=dict(
        n_steps_lookahead=10,
        trim_silence=True,
        gravity_compensation=True,
        reduced_action_space=True,
        control_timestep=0.05,
        wrong_press_termination=False,
        disable_fingering_reward=False,
        disable_forearm_reward=False,
        disable_colorization=False,
        disable_hand_collisions=False,
        primitive_fingertip_collisions=True,
      )
    )
    if record:
      print('Recording')
      env = robopianist_wrappers.PianoSoundVideoWrapper(
        environment=env,
        record_dir='./robopianist_recordings',
        record_every=1,
        camera_id='piano/back',
        height=100,
        width=100,
      )

    # cameras = ['piano/back', 'piano/closeup', 'piano/egocentric', 'piano/left', 'piano/right', 'piano/topdown']
    if render_image:
      cameras = ['piano/topdown']
      for camera in cameras:
        env = robopianist_wrappers.PixelWrapper(
          environment=env, render_kwargs=dict(camera_id=camera, height=128, width=128), observation_key=camera
        )

    env = robopianist_wrappers.MidiEvaluationWrapper(
      environment=env, deque_size=1
    )
    class LogMidiEvaluationWrapper(dm_env_wrappers.EnvironmentWrapper):
      def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        if timestep.last():
          metrics = self._environment.get_musical_metrics()
          f1_score = metrics["f1"]
        else:
          f1_score = 0
        timestep = timestep._replace(
          observation={**timestep.observation, "log_f1": f1_score}
        )
        return timestep

      def observation_spec(self):
        original_spec = super().observation_spec()
        original_spec["log_f1"] = dm_env.specs.Array(shape=(), dtype=np.float64, name="log_f1")
        return original_spec

    env = LogMidiEvaluationWrapper(env)
    env = dm_env_wrappers.EpisodeStatisticsWrapper(env)
    env = dm_env_wrappers.ObservationActionRewardWrapper(env)
    env = dm_env_wrappers.DmControlWrapper(env)

    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)  # TODO(js)
    self._env = embodied.wrappers.ExpandScalars(self._env)

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    return obs

  def render(self):
    return self._dmenv.physics.render(*self._size, camera_id=self._camera)
