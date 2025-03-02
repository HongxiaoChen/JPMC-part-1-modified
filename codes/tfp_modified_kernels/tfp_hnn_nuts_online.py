import collections
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.mcmc import kernel
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from . import hnn_leapfrog as leapfrog_impl

JAX_MODE = False

##############################################################
### BEGIN STATIC CONFIGURATION ###############################
##############################################################
TREE_COUNT_DTYPE = tf.int32           # Default: tf.int32

# Whether to use slice sampling (original NUTS implementation in [1]) or
# multinomial sampling (implementation in [2]) from the tree trajectory.
MULTINOMIAL_SAMPLE = False             # Default: True

# Whether to use U turn criteria in [1] or generalized U turn criteria in [2]
# to check the tree trajectory.
GENERALIZED_UTURN = False              # Default: True
##############################################################
### END STATIC CONFIGURATION #################################
##############################################################

__all__ = [
    'NoUTurnSampler',
]


class NUTSKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'NUTSKernelResults',
        [
            'target_log_prob',
            'grads_target_log_prob',
            'step_size',
            'log_accept_ratio',
            'leapfrogs_taken',  # How many leapfrogs each chain took this step.
            'leapfrogs_taken_trad',  # How many leapfrogs each chain took this step.
            'is_accepted',
            'reach_max_depth',
            'has_divergence',
            'energy',
            'seed',
            'use_traditional',           # New: whether to use traditional Hamiltonian
            'current_traditional_steps',  # New: number of steps in traditional Hamiltonian
            'hnn_error',                  # New: HNN error
            'used_traditional_in_this_tree'  # New: whether traditional Hamiltonian was used in this tree
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class MomentumStateSwap(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('MomentumStateSwap',
                           ['momentum_swap', 'state_swap'])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class OneStepMetaInfo(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('OneStepMetaInfo',
                           ['log_slice_sample',
                            'init_energy',
                            'write_instruction',
                            'read_instruction',
                           ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingState(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('TreeDoublingState',
                           ['momentum',
                            'state',
                            'target',
                            'target_grad_parts',
                            ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingStateCandidate(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TreeDoublingStateCandidate',
        [
            'state',
            'target',
            'target_grad_parts',
            'energy',
            'weight',
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class TreeDoublingMetaState(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'TreeDoublingMetaState',
        [
            'candidate_state',  # A namedtuple of TreeDoublingStateCandidate.
            'is_accepted',
            'momentum_sum',     # Sum of momentum of the current tree for
                                # generalized U turn criteria.
            'energy_diff_sum',  # Sum over all states explored within the
                                # subtree of Metropolis acceptance probabilities
                                # exp(min(0, H' - H0)), where H0 is the negative
                                # energy of the initial state and H' is the
                                # negative energy of a state explored in the
                                # subtree.
                                # TODO(b/150152798): Do sum in log-space.
            'leapfrog_count',   # How many leapfrogs each chain has taken.
            'leapfrog_count_trad',   # How many leapfrogs each chain has taken.
            'continue_tree',
            'not_divergence',
            'use_traditional',           # New: whether to use traditional Hamiltonian
            'current_traditional_steps',  # New: number of steps in traditional Hamiltonian
            'hnn_error',                  # New: HNN error
            'used_traditional_in_this_tree'  # New: whether traditional Hamiltonian was used in this tree
        ])):
  """Internal state and diagnostics for No-U-Turn Sampler."""
  __slots__ = ()


class NoUTurnSampler(kernel.TransitionKernel):
  """Runs one step of the No U-Turn Sampler.

  The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
  Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
  the curvature of the target density. Conceptually, one proposal consists of
  reversibly evolving a trajectory through the sample space, continuing until
  that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
  implements one random NUTS step from a given `current_state`.
  Mathematical details and derivations can be found in
  [Hoffman, Gelman (2011)][1] and [Betancourt (2018)][2].

  The `one_step` function can update multiple chains in parallel. It assumes
  that a prefix of leftmost dimensions of `current_state` index independent
  chain states (and are therefore updated independently).  The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions.  Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0][0, ...]` could have a
  different target distribution from `current_state[0][1, ...]`.  These
  semantics are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  """

  def __init__(self,
               hnn_model,
               hamiltonian_function,
               target_log_prob_fn,
               step_size,
               max_tree_depth=10,
               max_energy_diff=1000.,
               unrolled_leapfrog_steps=1,
               parallel_iterations=10,
               max_hnn_error_threshold=10.0,
               max_traditional_steps=20,
               experimental_shard_axis_names=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      max_tree_depth: Maximum depth of the tree implicitly built by NUTS. The
        maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
        the number of nodes in a binary tree `max_tree_depth` nodes deep. The
        default setting of 10 takes up to 1024 leapfrog steps.
      max_energy_diff: Scaler threshold of energy differences at each leapfrog,
        divergence samples are defined as leapfrog steps that exceed this
        threshold. Default to 1000.
      unrolled_leapfrog_steps: The number of leapfrogs to unroll per tree
        expansion step. Applies a direct linear multipler to the maximum
        trajectory length implied by max_tree_depth. Defaults to 1.
      parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.
      max_hnn_error_threshold: Threshold for the HNN error; if exceeded, switch
        to traditional Hamiltonian. Default to 10.0.
      max_traditional_steps: Maximum number of steps to use traditional
        Hamiltonian before switching back to HNN. Default to 20.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'NoUTurnSampler').
    """
    with tf.name_scope(name or 'NoUTurnSampler') as name:
      # Process `max_tree_depth` argument.
      max_tree_depth = tf.get_static_value(max_tree_depth)
      if max_tree_depth is None or max_tree_depth < 1:
        raise ValueError(
            'max_tree_depth must be known statically and >= 1 but was '
            '{}'.format(max_tree_depth))
      self._max_tree_depth = max_tree_depth

      # Compute parameters derived from `max_tree_depth`.
      instruction_array = build_tree_uturn_instruction(
          max_tree_depth, init_memory=-1)
      [
          write_instruction_numpy,
          read_instruction_numpy
      ] = generate_efficient_write_read_instruction(instruction_array)

      # TensorArray version of the read/write instruction need to be created
      # within the function call to be compatible with XLA. Here we store the
      # numpy version of the instruction and convert it to TensorArray later.
      self._write_instruction = write_instruction_numpy
      self._read_instruction = read_instruction_numpy

      # Process all other arguments.
      self._target_log_prob_fn = target_log_prob_fn
      self._hnn_model = hnn_model
      self._hamiltonian_function = hamiltonian_function
      self._step_size = step_size

      # New parameters
      self._max_hnn_error_threshold = tf.convert_to_tensor(
          max_hnn_error_threshold, dtype=tf.float32, name='max_hnn_error_threshold')
      self._max_traditional_steps = tf.convert_to_tensor(
          max_traditional_steps, dtype=tf.int32, name='max_traditional_steps')

      self._parameters = dict(
          hnn_model=hnn_model, 
          hamiltonian_function=hamiltonian_function,
          target_log_prob_fn=target_log_prob_fn,
          step_size=step_size,
          max_tree_depth=max_tree_depth,
          max_energy_diff=max_energy_diff,
          unrolled_leapfrog_steps=unrolled_leapfrog_steps,
          parallel_iterations=parallel_iterations,
          max_hnn_error_threshold=max_hnn_error_threshold,
          max_traditional_steps=max_traditional_steps,
          experimental_shard_axis_names=experimental_shard_axis_names,
          name=name,
      )
      self._parallel_iterations = parallel_iterations
      self._unrolled_leapfrog_steps = unrolled_leapfrog_steps
      self._name = name
      self._max_energy_diff = max_energy_diff

  @property
  def target_log_prob_fn(self):
    return self._target_log_prob_fn

  @property
  def hnn_model(self):
    return self._hnn_model

  @property
  def hamiltonian_function(self):
    return self._hamiltonian_function

  @property
  def step_size(self):
    return self._step_size

  @property
  def max_tree_depth(self):
    return self._max_tree_depth

  @property
  def max_energy_diff(self):
    return self._max_energy_diff

  @property
  def unrolled_leapfrog_steps(self):
    return self._unrolled_leapfrog_steps

  @property
  def name(self):
    return self._name

  @property
  def parallel_iterations(self):
    return self._parallel_iterations

  @property
  def write_instruction(self):
    return self._write_instruction

  @property
  def read_instruction(self):
    return self._read_instruction

  @property
  def parameters(self):
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results, seed=None):
    seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
    start_trajectory_seed, loop_seed = samplers.split_seed(seed)

    with tf.name_scope(self.name + '.one_step'):
      state_structure = current_state
      current_state = tf.nest.flatten(current_state)
      if (tf.nest.is_nested(state_structure)
          and (not mcmc_util.is_list_like(state_structure)
               or len(current_state) != len(state_structure))):
        # TODO(b/170865194): Support dictionaries and other non-list-like state.
        raise TypeError('NUTS does not currently support nested or '
                        'non-list-like state structures (saw: {}).'.format(
                            state_structure))

      current_target_log_prob = previous_kernel_results.target_log_prob
      [
          init_momentum,
          init_energy,
          log_slice_sample
      ] = self._start_trajectory_batched(current_state, current_target_log_prob,
                                         seed=start_trajectory_seed)

      # Get the previous state variables
      use_traditional = previous_kernel_results.use_traditional
      current_traditional_steps = previous_kernel_results.current_traditional_steps

      def _copy(v):
        return v * ps.ones(
            ps.pad(
                [2], paddings=[[0, ps.rank(v)]], constant_values=1),
            dtype=v.dtype)

      initial_state = TreeDoublingState(
          momentum=init_momentum,
          state=current_state,
          target=current_target_log_prob,
          target_grad_parts=previous_kernel_results.grads_target_log_prob)
      initial_step_state = tf.nest.map_structure(_copy, initial_state)

      if MULTINOMIAL_SAMPLE:
        init_weight = tf.zeros_like(init_energy)  # log(exp(H0 - H0))
      else:
        init_weight = tf.ones_like(init_energy, dtype=TREE_COUNT_DTYPE)

      candidate_state = TreeDoublingStateCandidate(
          state=current_state,
          target=current_target_log_prob,
          target_grad_parts=previous_kernel_results.grads_target_log_prob,
          energy=init_energy,
          weight=init_weight)

      initial_step_metastate = TreeDoublingMetaState(
          candidate_state=candidate_state,
          is_accepted=tf.zeros_like(init_energy, dtype=tf.bool),
          momentum_sum=init_momentum,
          energy_diff_sum=tf.zeros_like(init_energy),
          leapfrog_count=tf.zeros_like(init_energy, dtype=TREE_COUNT_DTYPE),
          leapfrog_count_trad=tf.zeros_like(init_energy, dtype=TREE_COUNT_DTYPE),
          continue_tree=tf.ones_like(init_energy, dtype=tf.bool),
          not_divergence=tf.ones_like(init_energy, dtype=tf.bool),
          use_traditional=use_traditional,
          current_traditional_steps=current_traditional_steps,
          hnn_error=tf.zeros_like(init_energy, dtype=tf.float32),
          used_traditional_in_this_tree=tf.zeros_like(init_energy, dtype=tf.bool)
      )

      # Convert the write/read instruction into TensorArray so that it is
      # compatible with XLA.
      write_instruction = tf.TensorArray(
          TREE_COUNT_DTYPE,
          size=len(self._write_instruction),
          clear_after_read=False).unstack(self._write_instruction)
      read_instruction = tf.TensorArray(
          tf.int32,
          size=len(self._read_instruction),
          clear_after_read=False).unstack(self._read_instruction)

      current_step_meta_info = OneStepMetaInfo(
          log_slice_sample=log_slice_sample,
          init_energy=init_energy,
          write_instruction=write_instruction,
          read_instruction=read_instruction
          )

      momentum_state_memory = MomentumStateSwap(
          momentum_swap=self.init_momentum_state_memory(init_momentum),
          state_swap=self.init_momentum_state_memory(current_state))

      step_size = _prepare_step_size(
          previous_kernel_results.step_size,
          current_target_log_prob.dtype,
          len(current_state))
      
      _, _, _, new_step_metastate = tf.while_loop(
          cond=lambda iter_, seed, state, metastate: (  # pylint: disable=g-long-lambda
              (iter_ < self.max_tree_depth) &
              tf.reduce_any(metastate.continue_tree)),
          body=lambda iter_, seed, state, metastate: self._loop_tree_doubling(  # pylint: disable=g-long-lambda
              step_size,
              momentum_state_memory,
              current_step_meta_info,
              iter_,
              state,
              metastate,
              seed),
          loop_vars=(
              tf.zeros([], dtype=tf.int32, name='iter'),
              loop_seed,
              initial_step_state,
              initial_step_metastate),
          parallel_iterations=self.parallel_iterations,
      )

      kernel_results = NUTSKernelResults(
          target_log_prob=new_step_metastate.candidate_state.target,
          grads_target_log_prob=(
              new_step_metastate.candidate_state.target_grad_parts),
          step_size=previous_kernel_results.step_size,
          log_accept_ratio=tf.math.log(
              new_step_metastate.energy_diff_sum /
              tf.cast(new_step_metastate.leapfrog_count,
                      dtype=new_step_metastate.energy_diff_sum.dtype)),
          leapfrogs_taken=(
              new_step_metastate.leapfrog_count * self.unrolled_leapfrog_steps
          ),
          leapfrogs_taken_trad=(
              new_step_metastate.leapfrog_count_trad * self.unrolled_leapfrog_steps
          ),
          is_accepted=new_step_metastate.is_accepted,
          reach_max_depth=new_step_metastate.continue_tree,
          has_divergence=~new_step_metastate.not_divergence,
          energy=new_step_metastate.candidate_state.energy,
          seed=seed,
          # New state variables
          use_traditional=new_step_metastate.use_traditional,
          current_traditional_steps=new_step_metastate.current_traditional_steps,
          hnn_error=new_step_metastate.hnn_error,
          used_traditional_in_this_tree=new_step_metastate.used_traditional_in_this_tree
      )

      result_state = tf.nest.pack_sequence_as(
          state_structure, new_step_metastate.candidate_state.state)
      return result_state, kernel_results

  def init_momentum_state_memory(self, input_tensors):
    """Allocate TensorArray for storing state and momentum."""
    shape_and_dtype = [(ps.shape(x_), x_.dtype) for x_ in input_tensors]
    return [  # pylint: disable=g-complex-comprehension
        ps.zeros(
            ps.concat([[max(self._write_instruction) + 1], s], axis=0),
            dtype=d) for (s, d) in shape_and_dtype
    ]

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    with tf.name_scope(self.name + '.bootstrap_results'):
      if not tf.nest.is_nested(init_state):
        init_state = [init_state]
      dummy_momentum = [tf.ones_like(state) for state in init_state]

      [
          _,
          _,
          current_target_log_prob,
          current_grads_log_prob,
      ] = leapfrog_impl.process_args(self.hnn_model, self.target_log_prob_fn, dummy_momentum,
                                   init_state)

      # Confirm that the step size is compatible with the state parts.
      _ = _prepare_step_size(
          self.step_size, current_target_log_prob.dtype, len(init_state))

      # Get batch_shape for initializing new state variables
      batch_shape = ps.shape(current_target_log_prob)

      return NUTSKernelResults(
          target_log_prob=current_target_log_prob,
          grads_target_log_prob=current_grads_log_prob,
          step_size=tf.nest.map_structure(
              lambda x: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
                  x,
                  dtype=current_target_log_prob.dtype,
                  name='step_size'),
              self.step_size),
          log_accept_ratio=tf.zeros_like(current_target_log_prob,
                                       name='log_accept_ratio'),
          leapfrogs_taken=tf.zeros_like(current_target_log_prob,
                                      dtype=TREE_COUNT_DTYPE,
                                      name='leapfrogs_taken'),
          leapfrogs_taken_trad=tf.zeros_like(current_target_log_prob,
                                      dtype=TREE_COUNT_DTYPE,
                                      name='leapfrogs_taken_trad'),
          is_accepted=tf.zeros_like(current_target_log_prob,
                                  dtype=tf.bool,
                                  name='is_accepted'),
          reach_max_depth=tf.zeros_like(current_target_log_prob,
                                      dtype=tf.bool,
                                      name='reach_max_depth'),
          has_divergence=tf.zeros_like(current_target_log_prob,
                                     dtype=tf.bool,
                                     name='has_divergence'),
          energy=compute_hamiltonian(
              current_target_log_prob, dummy_momentum,
              shard_axis_names=self.experimental_shard_axis_names),
          # Allow room for one_step's seed.
          seed=samplers.zeros_seed(),
          # New state variables
          use_traditional=tf.zeros_like(current_target_log_prob, 
                                        dtype=tf.bool, 
                                        name='use_traditional'),
          current_traditional_steps=tf.zeros_like(current_target_log_prob, 
                                                  dtype=tf.int32, 
                                                  name='current_traditional_steps'),
          hnn_error=tf.zeros_like(current_target_log_prob, dtype=tf.float32, name='hnn_error'),
          used_traditional_in_this_tree=tf.zeros_like(current_target_log_prob, 
                                                      dtype=tf.bool, 
                                                      name='used_traditional_in_this_tree')
      )

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)

  def _start_trajectory_batched(self, state, target_log_prob, seed):
    """Computations needed to start a trajectory."""
    with tf.name_scope('start_trajectory_batched'):
      seeds = list(samplers.split_seed(seed, n=len(state) + 1))
      momentum_seeds = distribute_lib.fold_in_axis_index(
          seeds[:-1], self.experimental_shard_axis_names)
      momentum = [
          samplers.normal(  # pylint: disable=g-complex-comprehension
              shape=ps.shape(x),
              dtype=x.dtype,
              seed=momentum_seeds[i]) for (i, x) in enumerate(state)
      ]
      init_energy = compute_hamiltonian(
          target_log_prob, momentum,
          shard_axis_names=self.experimental_shard_axis_names)

      if MULTINOMIAL_SAMPLE:
        return momentum, init_energy, None

      # Draw a slice variable u ~ Uniform(0, p(initial state, initial
      # momentum)) and compute log u. For numerical stability, we perform this
      # in log space where log u = log (u' * p(...)) = log u' + log
      # p(...) and u' ~ Uniform(0, 1).
      log_slice_sample = tf.math.log1p(-samplers.uniform(
          shape=ps.shape(init_energy),
          dtype=init_energy.dtype,
          seed=seeds[len(state)]))
      return momentum, init_energy, log_slice_sample

  def _loop_tree_doubling(self, step_size, momentum_state_memory,
                          current_step_meta_info, iter_, initial_step_state,
                          initial_step_metastate, seed):
    """Main loop for tree doubling."""
    with tf.name_scope('loop_tree_doubling'):
      (direction_seed,
       subtree_seed,
       acceptance_seed,
       next_seed) = samplers.split_seed(seed, n=4)
      batch_shape = ps.shape(current_step_meta_info.init_energy)
      direction = tf.cast(
          samplers.uniform(
              shape=batch_shape,
              minval=0,
              maxval=2,
              dtype=tf.int32,
              seed=direction_seed),
          dtype=tf.bool)

      tree_start_states = tf.nest.map_structure(
          lambda v: bu.where_left_justified_mask(direction, v[1], v[0]),
          initial_step_state)

      directions_expanded = [
          bu.left_justified_expand_dims_like(direction, state)
          for state in tree_start_states.state
      ]

      integrator_hnn = leapfrog_impl.HNNLeapfrogIntegrator(
          self.hnn_model,
          self.target_log_prob_fn,
          step_sizes=[
              tf.where(d, ss, -ss)
              for d, ss in zip(directions_expanded, step_size)
          ],
          num_steps=self.unrolled_leapfrog_steps)

      integrator_trad = leapfrog_impl.HNNLeapfrogIntegrator(
          self.hamiltonian_function,
          self.target_log_prob_fn,
          step_sizes=[
              tf.where(d, ss, -ss)
              for d, ss in zip(directions_expanded, step_size)
          ],
          num_steps=self.unrolled_leapfrog_steps)      
      

      [
          candidate_tree_state,
          tree_final_states,
          final_not_divergence,
          continue_tree_final,
          energy_diff_tree_sum,
          momentum_subtree_cumsum,
          leapfrogs_taken,
          leapfrogs_taken_trad,
          new_use_traditional,
          new_current_traditional_steps,
          hnn_error,
          used_traditional_in_this_tree
      ] = self._build_sub_tree(
          directions_expanded,
          integrator_hnn,
          integrator_trad,
          current_step_meta_info,
          # num_steps_at_this_depth = 2**iter_ = 1 << iter_
          tf.bitwise.left_shift(1, iter_),
          tree_start_states,
          initial_step_metastate.continue_tree,
          initial_step_metastate.not_divergence,
          momentum_state_memory,
          initial_step_metastate.use_traditional,
          initial_step_metastate.current_traditional_steps,
          seed=subtree_seed)

      last_candidate_state = initial_step_metastate.candidate_state

      energy_diff_sum = (
          energy_diff_tree_sum + initial_step_metastate.energy_diff_sum)
      if MULTINOMIAL_SAMPLE:
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.constant(-np.inf, dtype=candidate_tree_state.weight.dtype))
        weight_sum = generic.log_add_exp(
            tree_weight, last_candidate_state.weight)
        log_accept_thresh = tree_weight - last_candidate_state.weight
      else:
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.zeros([], dtype=TREE_COUNT_DTYPE))
        weight_sum = tree_weight + last_candidate_state.weight
        log_accept_thresh = tf.math.log(
            tf.cast(tree_weight, tf.float32) /
            tf.cast(last_candidate_state.weight, tf.float32))
        
      log_accept_thresh = tf.where(
          tf.math.is_nan(log_accept_thresh),
          tf.zeros([], log_accept_thresh.dtype),
          log_accept_thresh)
      u = tf.math.log1p(-samplers.uniform(
          shape=batch_shape,
          dtype=log_accept_thresh.dtype,
          seed=acceptance_seed))
      is_sample_accepted = u <= log_accept_thresh

      choose_new_state = is_sample_accepted & continue_tree_final

      new_candidate_state = TreeDoublingStateCandidate(
          state=[
              bu.where_left_justified_mask(choose_new_state, s0, s1)
              for s0, s1 in zip(candidate_tree_state.state,
                                last_candidate_state.state)
          ],
          target=bu.where_left_justified_mask(
              choose_new_state,
              candidate_tree_state.target,
              last_candidate_state.target),
          target_grad_parts=[
              bu.where_left_justified_mask(choose_new_state, grad0, grad1)
              for grad0, grad1 in zip(candidate_tree_state.target_grad_parts,
                                      last_candidate_state.target_grad_parts)
          ],
          energy=bu.where_left_justified_mask(
              choose_new_state,
              candidate_tree_state.energy,
              last_candidate_state.energy),
          weight=weight_sum)

      for new_candidate_state_temp, old_candidate_state_temp in zip(
          new_candidate_state.state, last_candidate_state.state):
        tensorshape_util.set_shape(new_candidate_state_temp,
                                   old_candidate_state_temp.shape)

      for new_candidate_grad_temp, old_candidate_grad_temp in zip(
          new_candidate_state.target_grad_parts,
          last_candidate_state.target_grad_parts):
        tensorshape_util.set_shape(new_candidate_grad_temp,
                                   old_candidate_grad_temp.shape)

      # Update left right information of the trajectory, and check trajectory
      # level U turn
      tree_otherend_states = tf.nest.map_structure(
          lambda v: bu.where_left_justified_mask(direction, v[0], v[1]),
          initial_step_state)

      new_step_state = tf.nest.pack_sequence_as(initial_step_state, [
          tf.stack([  # pylint: disable=g-complex-comprehension
              bu.where_left_justified_mask(direction, right, left),
              bu.where_left_justified_mask(direction, left, right),
          ], axis=0)
          for left, right in zip(tf.nest.flatten(tree_final_states),
                                 tf.nest.flatten(tree_otherend_states))
      ])

      momentum_tree_cumsum = []
      for p0, p1 in zip(
          initial_step_metastate.momentum_sum, momentum_subtree_cumsum):
        momentum_part_temp = p0 + p1
        tensorshape_util.set_shape(momentum_part_temp, p0.shape)
        momentum_tree_cumsum.append(momentum_part_temp)

      for new_state_temp, old_state_temp in zip(
          tf.nest.flatten(new_step_state),
          tf.nest.flatten(initial_step_state)):
        tensorshape_util.set_shape(new_state_temp, old_state_temp.shape)

      if GENERALIZED_UTURN:
        state_diff = momentum_tree_cumsum
      else:
        state_diff = [s[1] - s[0] for s in new_step_state.state]

      no_u_turns_trajectory = has_not_u_turn(
          state_diff,
          [m[0] for m in new_step_state.momentum],
          [m[1] for m in new_step_state.momentum],
          log_prob_rank=ps.rank_from_shape(batch_shape),
          shard_axis_names=self.experimental_shard_axis_names)

      # After building the tree, update the used_traditional_in_this_tree 
      # based on whether traditional NUTS was used in this tree
      final_used_traditional = tf.logical_or(
          initial_step_metastate.used_traditional_in_this_tree,
          used_traditional_in_this_tree
      )
      
      # If traditional NUTS was used in this tree, increment the counter; 
      # if not used, reset the counter
      updated_current_traditional_steps = tf.where(
          final_used_traditional,
          initial_step_metastate.current_traditional_steps + 1,
          tf.zeros_like(initial_step_metastate.current_traditional_steps)
      )
      
      # Check if the maximum number of steps has been reached, if so, reset the state
      reset_condition = updated_current_traditional_steps >= self._max_traditional_steps
      
      final_use_traditional = tf.where(
          reset_condition,
          tf.zeros_like(final_used_traditional),
          new_use_traditional
      )
      
      final_current_traditional_steps = tf.where(
          reset_condition,
          tf.zeros_like(updated_current_traditional_steps),
          updated_current_traditional_steps
      )

      new_step_metastate = TreeDoublingMetaState(
          candidate_state=new_candidate_state,
          is_accepted=choose_new_state | initial_step_metastate.is_accepted,
          momentum_sum=momentum_tree_cumsum,
          energy_diff_sum=energy_diff_sum,
          continue_tree=continue_tree_final & no_u_turns_trajectory,
          not_divergence=final_not_divergence,
          leapfrog_count=(initial_step_metastate.leapfrog_count +
                          leapfrogs_taken),
          leapfrog_count_trad=(initial_step_metastate.leapfrog_count_trad +
                              leapfrogs_taken_trad),
          use_traditional=final_use_traditional,
          current_traditional_steps=final_current_traditional_steps,
          hnn_error=hnn_error,
          used_traditional_in_this_tree=final_used_traditional
      )

      return iter_ + 1, next_seed, new_step_state, new_step_metastate

  def _build_sub_tree(self,
                    directions,
                    integrator_hnn,
                    integrator_trad,
                    current_step_meta_info,
                    nsteps,
                    initial_state,
                    continue_tree,
                    not_divergence,
                    momentum_state_memory,
                    use_traditional,
                    current_traditional_steps,
                    seed,
                    name=None):
    with tf.name_scope('build_sub_tree'):
      batch_shape = ps.shape(current_step_meta_info.init_energy)
      # We never want to select the initial state
      if MULTINOMIAL_SAMPLE:
        init_weight = tf.fill(
            batch_shape,
            tf.constant(-np.inf,
                        dtype=current_step_meta_info.init_energy.dtype))
      else:
        init_weight = tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE)

      init_momentum_cumsum = [tf.zeros_like(x) for x in initial_state.momentum]
      initial_state_candidate = TreeDoublingStateCandidate(
          state=initial_state.state,
          target=initial_state.target,
          target_grad_parts=initial_state.target_grad_parts,
          energy=initial_state.target,
          weight=init_weight)
      energy_diff_sum = tf.zeros_like(current_step_meta_info.init_energy,
                                      name='energy_diff_sum')
      # Initialize used_traditional_in_this_tree to False
      used_traditional_in_this_tree = tf.zeros_like(current_step_meta_info.init_energy, 
                                                  dtype=tf.bool, 
                                                  name='used_traditional_in_this_tree')
      [
          _,
          _,
          energy_diff_tree_sum,
          momentum_tree_cumsum,
          leapfrogs_taken,
          leapfrogs_taken_trad,
          final_state,
          candidate_tree_state,
          final_continue_tree,
          final_not_divergence,
          momentum_state_memory,
          new_use_traditional,
          new_current_traditional_steps,
          hnn_error,
          used_traditional_in_this_tree
      ] = tf.while_loop(
          cond=lambda iter_, seed, energy_diff_sum, init_momentum_cumsum,  # pylint: disable=g-long-lambda
                      leapfrogs_taken, leapfrogs_taken_trad, state, state_c, continue_tree,
                      not_divergence, momentum_state_memory, use_trad, trad_steps, hnn_err, used_trad_in_tree: (
                          (iter_ < nsteps) & tf.reduce_any(continue_tree)),
          body=lambda iter_, seed, energy_diff_sum, init_momentum_cumsum,  # pylint: disable=g-long-lambda
                      leapfrogs_taken, leapfrogs_taken_trad, state, state_c, continue_tree,
                      not_divergence, momentum_state_memory, use_trad, trad_steps, hnn_err, used_trad_in_tree: (
                          self._loop_build_sub_tree(
                              directions, integrator_hnn, integrator_trad, current_step_meta_info,
                              iter_, energy_diff_sum, init_momentum_cumsum,
                              leapfrogs_taken, leapfrogs_taken_trad, state, state_c, continue_tree,
                              not_divergence, momentum_state_memory, seed, use_trad, trad_steps, used_trad_in_tree)),
          loop_vars=(
              tf.zeros([], dtype=tf.int32, name='iter'),
              seed,
              energy_diff_sum,
              init_momentum_cumsum,
              tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE),
              tf.zeros(batch_shape, dtype=TREE_COUNT_DTYPE),
              initial_state,
              initial_state_candidate,
              continue_tree,
              not_divergence,
              momentum_state_memory,
              use_traditional,
              current_traditional_steps,
              tf.zeros_like(current_step_meta_info.init_energy, dtype=tf.float32),
              used_traditional_in_this_tree
          ),
          parallel_iterations=self.parallel_iterations
      )

    return (
        candidate_tree_state,
        final_state,
        final_not_divergence,
        final_continue_tree,
        energy_diff_tree_sum,
        momentum_tree_cumsum,
        leapfrogs_taken,
        leapfrogs_taken_trad,
        new_use_traditional,
        new_current_traditional_steps,
        hnn_error,
        used_traditional_in_this_tree
    )

  def _loop_build_sub_tree(self,
                           directions,
                           integrator_hnn,
                           integrator_trad,
                           current_step_meta_info,
                           iter_,
                           energy_diff_sum_previous,
                           momentum_cumsum_previous,
                           leapfrogs_taken,
                           leapfrogs_taken_trad,
                           prev_tree_state,
                           candidate_tree_state,
                           continue_tree_previous,
                           not_divergent_previous,
                           momentum_state_memory,
                           seed,
                           use_traditional,
                           current_traditional_steps,
                           used_traditional_in_this_tree_previous):
    """Base case in tree doubling."""
    acceptance_seed, next_seed = samplers.split_seed(seed)
    with tf.name_scope('loop_build_sub_tree'):
      # First calculate the result of HNN, for error checking
      [
          hnn_momentum_parts,
          hnn_state_parts,
          hnn_target,
          hnn_target_grad_parts
      ] = integrator_hnn(prev_tree_state.momentum,
                     prev_tree_state.state,
                     prev_tree_state.target,
                     prev_tree_state.target_grad_parts)

      # Calculate the energy of HNN (negative Hamiltonian)
      hnn_energy = compute_hamiltonian(
          hnn_target, hnn_momentum_parts,
          shard_axis_names=self.experimental_shard_axis_names)
      
      # Calculate the energy difference
      hnn_energy_diff = hnn_energy - current_step_meta_info.init_energy
      
      # Calculate the HNN error
      # Note: In the code, energy = -Hamiltonian, 
      # log_slice_sample comes from log(1 - Uniform(0,1))
      # The error check is: log_slice_sample - hnn_energy_diff
      hnn_error = current_step_meta_info.log_slice_sample - hnn_energy_diff
      error_exceeds_threshold = hnn_error > self._max_hnn_error_threshold
      
      # If the HNN error exceeds the threshold, 
      # or the traditional NUTS is already being used and the step count has not reached the maximum,
      # use the traditional NUTS
      should_use_traditional = tf.logical_or(
          error_exceeds_threshold,
          tf.logical_and(use_traditional, 
                        current_traditional_steps < self._max_traditional_steps)
      )
      
      # Update used_traditional_in_this_tree, 
      # if the current step uses traditional NUTS, 
      # then the whole tree is marked as using traditional NUTS
      used_traditional_in_this_tree = tf.logical_or(
          used_traditional_in_this_tree_previous,
          should_use_traditional
      )
      
      # Use tf.cond to select integrator
      def use_hnn():
          return hnn_momentum_parts, hnn_state_parts, hnn_target, hnn_target_grad_parts
      
      def use_trad():
          return integrator_trad(prev_tree_state.momentum,
                                prev_tree_state.state,
                                prev_tree_state.target,
                                prev_tree_state.target_grad_parts)
      
      next_momentum_parts, next_state_parts, next_target, next_target_grad_parts = tf.cond(
          should_use_traditional,
          use_trad,
          use_hnn
      )
      
      # Only after building the tree, 
      # decide whether to increment current_traditional_steps
      new_current_traditional_steps = current_traditional_steps
      new_use_traditional = should_use_traditional

      next_tree_state = TreeDoublingState(
          momentum=next_momentum_parts,
          state=next_state_parts,
          target=next_target,
          target_grad_parts=next_target_grad_parts)
      momentum_cumsum = [p0 + p1 for p0, p1 in zip(momentum_cumsum_previous,
                                                   next_momentum_parts)]
      
      # If the tree has not yet terminated previously, we count this leapfrog.
      leapfrogs_taken = tf.where(
          continue_tree_previous, leapfrogs_taken + 1, leapfrogs_taken)
      
      leapfrogs_taken_trad = tf.where(
          continue_tree_previous & should_use_traditional, leapfrogs_taken_trad + 1, leapfrogs_taken_trad)

      write_instruction = current_step_meta_info.write_instruction
      read_instruction = current_step_meta_info.read_instruction
      init_energy = current_step_meta_info.init_energy

      if GENERALIZED_UTURN:
        state_to_write = momentum_cumsum_previous
        state_to_check = momentum_cumsum
      else:
        state_to_write = next_state_parts
        state_to_check = next_state_parts

      batch_shape = ps.shape(next_target)
      has_not_u_turn_init = ps.ones(batch_shape, dtype=tf.bool)

      read_index = read_instruction.gather([iter_])[0]
      no_u_turns_within_tree = has_not_u_turn_at_all_index(  # pylint: disable=g-long-lambda
          read_index,
          directions,
          momentum_state_memory,
          next_momentum_parts,
          state_to_check,
          has_not_u_turn_init,
          log_prob_rank=ps.rank(next_target),
          shard_axis_names=self.experimental_shard_axis_names)

      # Get index to write state into memory swap
      write_index = write_instruction.gather([iter_])
      momentum_state_memory = MomentumStateSwap(
          momentum_swap=[
              _safe_tensor_scatter_nd_update(old, [write_index], [new])
              for old, new in zip(momentum_state_memory.momentum_swap,
                                  next_momentum_parts)
          ],
          state_swap=[
              _safe_tensor_scatter_nd_update(old, [write_index], [new])
              for old, new in zip(momentum_state_memory.state_swap,
                                  state_to_write)
          ])

      energy = compute_hamiltonian(
          next_target, next_momentum_parts,
          shard_axis_names=self.experimental_shard_axis_names)
      current_energy = tf.where(tf.math.is_nan(energy),
                                tf.constant(-np.inf, dtype=energy.dtype),
                                energy)
      energy_diff = current_energy - init_energy

      if MULTINOMIAL_SAMPLE:
        not_divergent = -energy_diff < self.max_energy_diff
        weight_sum = generic.log_add_exp(
            candidate_tree_state.weight, energy_diff)
        log_accept_thresh = energy_diff - weight_sum
      else:
        log_slice_sample = current_step_meta_info.log_slice_sample
        not_divergent = log_slice_sample - energy_diff < self.max_energy_diff
        # Uniform sampling on the trajectory within the subtree across valid
        # samples.
        is_valid = log_slice_sample <= energy_diff
        weight_sum = tf.where(is_valid,
                              candidate_tree_state.weight + 1,
                              candidate_tree_state.weight)
        log_accept_thresh = tf.where(
            is_valid,
            -tf.math.log(tf.cast(weight_sum, dtype=tf.float32)),
            tf.constant(-np.inf, dtype=tf.float32))
      u = tf.math.log1p(-samplers.uniform(
          shape=batch_shape,
          dtype=log_accept_thresh.dtype,
          seed=acceptance_seed))
      is_sample_accepted = u <= log_accept_thresh

      next_candidate_tree_state = TreeDoublingStateCandidate(
          state=[
              bu.where_left_justified_mask(is_sample_accepted, s0, s1)
              for s0, s1 in zip(next_state_parts, candidate_tree_state.state)
          ],
          target=bu.where_left_justified_mask(
              is_sample_accepted, next_target, candidate_tree_state.target),
          target_grad_parts=[
              bu.where_left_justified_mask(is_sample_accepted, grad0, grad1)
              for grad0, grad1 in zip(next_target_grad_parts,
                                      candidate_tree_state.target_grad_parts)
          ],
          energy=bu.where_left_justified_mask(
              is_sample_accepted,
              current_energy,
              candidate_tree_state.energy),
          weight=weight_sum)

      continue_tree = not_divergent & continue_tree_previous
      continue_tree_next = no_u_turns_within_tree & continue_tree

      not_divergent_tokeep = tf.where(
          continue_tree_previous,
          not_divergent,
          ps.ones(batch_shape, dtype=tf.bool))

      # min(1., exp(energy_diff)).
      exp_energy_diff = tf.math.exp(tf.minimum(energy_diff, 0.))
      energy_diff_sum = tf.where(continue_tree,
                                 energy_diff_sum_previous + exp_energy_diff,
                                 energy_diff_sum_previous)

      return (
          iter_ + 1,
          next_seed,
          energy_diff_sum,
          momentum_cumsum,
          leapfrogs_taken,
          leapfrogs_taken_trad,
          next_tree_state,
          next_candidate_tree_state,
          continue_tree_next,
          not_divergent_previous & not_divergent_tokeep,
          momentum_state_memory,
          new_use_traditional,
          new_current_traditional_steps,
          hnn_error,
          used_traditional_in_this_tree
      )


def has_not_u_turn_at_all_index(read_indexes, direction, momentum_state_memory,
                                momentum_right, state_right,
                                no_u_turns_within_tree, log_prob_rank,
                                shard_axis_names=None):
  """Check u turn for early stopping."""

  def _get_left_state_and_check_u_turn(left_current_index, no_u_turns_last):
    """Check U turn on a single index."""
    momentum_left = [
        tf.gather(x, left_current_index, axis=0)
        for x in momentum_state_memory.momentum_swap
    ]
    state_left = [
        tf.gather(x, left_current_index, axis=0)
        for x in momentum_state_memory.state_swap
    ]
    # Note that in generalized u turn, state_diff is actually the cumulated sum
    # of the momentum.
    state_diff = [s1 - s2 for s1, s2 in zip(state_right, state_left)]
    if not GENERALIZED_UTURN:
      state_diff = [tf.where(d, m, -m) for d, m in zip(direction, state_diff)]

    no_u_turns_current = has_not_u_turn(
        state_diff,
        momentum_left,
        momentum_right,
        log_prob_rank,
        shard_axis_names=shard_axis_names)
    return left_current_index + 1, no_u_turns_current & no_u_turns_last

  _, no_u_turns_within_tree = tf.while_loop(
      cond=lambda i, no_u_turn: ((i < read_indexes[1]) &  # pylint: disable=g-long-lambda
                                 tf.reduce_any(no_u_turn)),
      body=_get_left_state_and_check_u_turn,
      loop_vars=(read_indexes[0], no_u_turns_within_tree))
  return no_u_turns_within_tree


def has_not_u_turn(state_diff,
                   momentum_left,
                   momentum_right,
                   log_prob_rank,
                   shard_axis_names=None):
  """If the trajectory does not exhibit a U-turn pattern."""
  shard_axis_names = (shard_axis_names or ([None] * len(state_diff)))
  def reduce_sum(x, m, shard_axes):
    out = tf.reduce_sum(x, axis=ps.range(log_prob_rank, ps.rank(m)))
    if shard_axes is not None:
      out = distribute_lib.psum(out, shard_axes)
    return out
  with tf.name_scope('has_not_u_turn'):
    batch_dot_product_left = sum(
        reduce_sum(s_diff * m, m, axes)
        for s_diff, m, axes in zip(state_diff, momentum_left,
                                   shard_axis_names)
    )
    batch_dot_product_right = sum(
        reduce_sum(s_diff * m, m, axes)
        for s_diff, m, axes in zip(state_diff, momentum_right,
                                   shard_axis_names)
    )
    return (batch_dot_product_left >= 0) & (batch_dot_product_right >= 0)


def build_tree_uturn_instruction(max_depth, init_memory=0):
  """Run build tree and output the u turn checking input instruction."""

  def _buildtree(address, depth):
    if depth == 0:
      address += 1
      return address, address
    else:
      address_left, address_right = _buildtree(address, depth - 1)
      _, address_right = _buildtree(address_right, depth - 1)
      instruction.append((address_left, address_right))
      return address_left, address_right

  instruction = []
  _, _ = _buildtree(init_memory, max_depth)
  return np.unique(np.array(instruction, dtype=np.int32), axis=0)


def generate_efficient_write_read_instruction(instruction_array):
  """Statically generate a memory efficient write/read instruction."""
  nsteps_within_tree = np.max(instruction_array) + 1
  instruction_mat = np.zeros((nsteps_within_tree, nsteps_within_tree))
  for previous_step, current_step in instruction_array:
    instruction_mat[previous_step, current_step] = 1

  # Generate a sparse matrix that represents the memory footprint:
  #   -1 : no need to save to memory (these are odd steps)
  #    1 : needed for check u turn (either already in memory or will be saved)
  #    0 : still in memory but not needed for check u turn
  for i in range(nsteps_within_tree):
    temp = instruction_mat[i]
    endpoint = np.where(temp == 1)[0]
    if endpoint.size > 0:
      temp[:i] = -1
      temp[endpoint[-1]+1:] = -1
      instruction_mat[i] = temp
    else:
      instruction_mat[i] = -1

  # In the classical U-turn check, the writing is only at odd step and the
  # instruction follows squence A000120 (https://oeis.org/A000120)
  to_write_temp = np.sum(instruction_mat > -1, axis=0)
  write_instruction = to_write_temp - 1
  write_instruction[np.diag(instruction_mat) == -1] = max(to_write_temp)

  read_instruction = []
  for i in range(nsteps_within_tree):
    temp_instruction = instruction_mat[:, i]
    if np.sum(temp_instruction == 1) > 0:
      r = np.where(temp_instruction[temp_instruction >= 0] == 1)[0][0]
      read_instruction.append([r, r + np.sum(temp_instruction == 1)])
    else:
      # If there is no instruction to do U turn check (e.g., odd step in the
      # original U turn check scheme), we append a pair of 0s as instruction.
      # In the inner most while loop of build tree, we loop through the read
      # instruction to check U turn - looping from 0 to 0 works with the
      # existing code while no computation happens.
      read_instruction.append([0, 0])
  return write_instruction, np.asarray(read_instruction)


def _prepare_step_size(step_size, dtype, n_state_parts):
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=dtype, name='step_size')
  if len(step_sizes) == 1:
    step_sizes *= n_state_parts
  if n_state_parts != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  return step_sizes


def compute_hamiltonian(target_log_prob, momentum_parts,
                        shard_axis_names=None):
  """Compute the Hamiltonian of the current system."""
  shard_axis_names = (shard_axis_names or ([None] * len(momentum_parts)))
  independent_chain_ndims = ps.rank(target_log_prob)
  def compute_sum_sq(v, shard_axes):
    sum_sq = tf.reduce_sum(v ** 2., axis=ps.range(
        independent_chain_ndims, ps.rank(v)))
    if shard_axes is not None:
      sum_sq = distribute_lib.psum(sum_sq, shard_axes)
    return sum_sq
  momentum_sq_parts = (
      tf.cast(  # pylint: disable=g-complex-comprehension
          compute_sum_sq(m, axes),
          dtype=target_log_prob.dtype)
      for m, axes in zip(momentum_parts, shard_axis_names))
  # TODO(jvdillon): Verify no broadcasting happening.
  return target_log_prob - 0.5 * sum(momentum_sq_parts)


def _safe_tensor_scatter_nd_update(tensor, indices, updates):
  if tensorshape_util.num_elements(tensor.shape) == 0:
    return tensor
  return tf.tensor_scatter_nd_update(tensor, indices, updates)
