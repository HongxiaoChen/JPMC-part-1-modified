import collections
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from .hnn_leapfrog import HNNLeapfrogIntegrator

__all__ = [
    'HNNHMC',
    'UncalibratedHNNHMC',
    'UncalibratedHNNHMCKernelResults',
]

class UncalibratedHNNHMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedHNNHMCKernelResults',
        [
            'log_acceptance_correction',  
            'target_log_prob',           # target log probability
            'initial_momentum',          # initial momentum
            'final_momentum',            # final momentum
            'step_size',                 # step size
            'num_leapfrog_steps',        # Leapfrog steps
            'seed'                      # seed

        ])):
    """Internal state and diagnostic information for HNN-HMC."""
    __slots__ = ()


class HNNHMCExtraKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple('HNNHMCExtraKernelResults',
                          ['step_size_assign'])):
    """Extra kernel results for HNN-HMC."""
    __slots__ = ()


class UncalibratedHNNHMC(kernel_base.TransitionKernel):
    """Uncalibrated HMC kernel using HNN."""

    def __init__(self,
                 step_size,
                 num_leapfrog_steps,
                 hnn_model,
                 target_fn,
                 state_gradients_are_stopped=False,
                 store_parameters_in_results=False,
                 name=None):
        """Initialize HNN-HMC kernel."""
        self._parameters = dict(
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            hnn_model=hnn_model,
            target_fn=target_fn,
            state_gradients_are_stopped=state_gradients_are_stopped,
            store_parameters_in_results=store_parameters_in_results,
            name=name or 'hnn_hmc_kernel'
        )


    @property
    def step_size(self):
        return self._parameters['step_size']

    @property
    def num_leapfrog_steps(self):
        return self._parameters['num_leapfrog_steps']

    @property
    def hnn_model(self):
        return self._parameters['hnn_model']

    @property
    def state_gradients_are_stopped(self):
        return self._parameters['state_gradients_are_stopped']

    @property
    def store_parameters_in_results(self):
        return self._parameters['store_parameters_in_results']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    @property
    def target_fn(self):
        return self._parameters['target_fn']

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Execute one step of HNN-HMC."""
        with tf.name_scope(mcmc_util.make_name(self.name, 'hnn_hmc', 'one_step')):
            # Prepare arguments
            current_state_parts, step_sizes = self._prepare_args(
                current_state, self.step_size)

            # Sample new momentum
            seed = samplers.sanitize_seed(seed)
            momentum_parts = []
            for s in current_state_parts:
                momentum_parts.append(samplers.normal(
                    shape=tf.shape(s),
                    seed=seed))

            # Use HNN integrator
            integrator = HNNLeapfrogIntegrator(
                self.hnn_model,
                self.target_fn,
                step_sizes,
                self.num_leapfrog_steps)

            [
                next_momentum_parts,
                next_state_parts,
                new_target,
                _
            ] = integrator(momentum_parts, current_state_parts)

            independent_chain_ndims = ps.rank(new_target)

            # Create new kernel results
            kernel_results = UncalibratedHNNHMCKernelResults(
                log_acceptance_correction=_compute_log_acceptance_correction(
                    momentum_parts, next_momentum_parts,
                    independent_chain_ndims,
              shard_axis_names=self.experimental_shard_axis_names), # come back to modify later
                target_log_prob=new_target, # target log probability
                initial_momentum=momentum_parts,
                final_momentum=next_momentum_parts,
                step_size=step_sizes,
                num_leapfrog_steps=self.num_leapfrog_steps,
                seed=seed)
            def maybe_flatten(x):
                    return x if mcmc_util.is_list_like(current_state) else x[0]

            return maybe_flatten(next_state_parts), kernel_results

    def bootstrap_results(self, init_state):
        """Create initial kernel results."""
        with tf.name_scope(mcmc_util.make_name(
            self.name, 'hnn_hmc', 'bootstrap_results')):
            init_state_parts, _ = mcmc_util.prepare_state_parts(init_state)
            
            if self.state_gradients_are_stopped:
                init_state_parts = [tf.stop_gradient(x) for x in init_state_parts]

            # make sure correct dimension
            init_momentum = [tf.zeros_like(x) for x in init_state_parts]
            
            # make sure concat has correct dimension
            q = init_state_parts[0]  # shape: [batch_size]
            p = init_momentum[0]     # shape: [batch_size]
                
            z = tf.concat([q, p], axis=-1)  # shape: [batch_size, 2]
            init_target = self.target_fn(q)  # shape: [batch_size]

            results = UncalibratedHNNHMCKernelResults(  
                log_acceptance_correction=tf.zeros_like(init_target), 
                target_log_prob=init_target,
                initial_momentum=init_momentum,
                final_momentum=init_momentum,
                step_size=[],
                num_leapfrog_steps=[],
                seed=samplers.zeros_seed()
            )

            if self.store_parameters_in_results:
                results = results._replace(
                    step_size=tf.nest.map_structure(
                        lambda x: tf.convert_to_tensor(
                            x, dtype=tf.float32, name='step_size'),
                        self.step_size),
                    num_leapfrog_steps=tf.convert_to_tensor(
                        self.num_leapfrog_steps, dtype=tf.int32,
                        name='num_leapfrog_steps'))

            return results

    def _prepare_args(self, state, step_size):
        """Simplified argument processing."""
        state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
        if self.state_gradients_are_stopped:
            state_parts = [tf.stop_gradient(x) for x in state_parts]
        
        step_sizes, _ = mcmc_util.prepare_state_parts(
            step_size, dtype=tf.float32, name='step_size')
        if len(step_sizes) == 1:
            step_sizes *= len(state_parts)
            
        return state_parts, step_sizes


class HNNHMC(kernel_base.TransitionKernel):
    """
    HMC sampler using HNN. 
    It is a wrapper of UncalibratedHNNHMC, through Metropolis-Hastings.
    """
    
    def __init__(self,
                 step_size,
                 num_leapfrog_steps,
                 hnn_model,
                 target_fn,
                 state_gradients_are_stopped=False,
                 store_parameters_in_results=False,
                 name=None):
        """Initialize HNN-HMC sampler."""
        self._parameters = dict(
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            hnn_model=hnn_model,
            target_fn=target_fn,
            state_gradients_are_stopped=state_gradients_are_stopped,
            store_parameters_in_results=store_parameters_in_results,
            name=name or 'hnn_hmc'
        )
        
        # Create uncalibrated HNN-HMC kernel
        self._impl = UncalibratedHNNHMC(
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            hnn_model=hnn_model,
            target_fn=target_fn,
            state_gradients_are_stopped=state_gradients_are_stopped,
            store_parameters_in_results=store_parameters_in_results,
            name=name)
        
        # Wrap with Metropolis-Hastings algorithm
        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=self._impl,
            name=name)

    @property
    def step_size(self):
        return self._parameters['step_size']

    @property
    def num_leapfrog_steps(self):
        return self._parameters['num_leapfrog_steps']

    @property
    def hnn_model(self):
        return self._parameters['hnn_model']

    @property
    def state_gradients_are_stopped(self):
        return self._parameters['state_gradients_are_stopped']

    @property
    def store_parameters_in_results(self):
        return self._parameters['store_parameters_in_results']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        """Return a dict of parameters."""
        return self._parameters

    @property
    def is_calibrated(self):
        return True

    @property
    def target_fn(self):
        return self._parameters['target_fn']

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Execute one step of HMC sampling."""
        return self._impl.one_step(current_state, previous_kernel_results, seed=seed)

    def bootstrap_results(self, init_state):
        """Initialize kernel results."""
        return self._impl.bootstrap_results(init_state)
    
def _compute_log_acceptance_correction(current_momentums,
                                       proposed_momentums,
                                       independent_chain_ndims,
                                       shard_axis_names=None,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.

  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:

  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```

  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.

  Inserting this into the detailed balance equation implies:

  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```

  One definition of `a(x'|x)` which satisfies (*) is:

  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```

  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)

  We call the bracketed term the "acceptance correction".

  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Assuming a standard Gaussian distribution for momentums, the chain eventually
  converges to:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  ```

  Relating this back to Metropolis-Hastings parlance, for HMC we have:

  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```

  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:

  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [exp(-0.5 z**2) / exp(-0.5 z'**2)]
                       target_prob(x)
  ```

  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)

  Args:
    current_momentums: `Tensor` representing the value(s) of the current
      momentum(s) of the state (parts).
    proposed_momentums: `Tensor` representing the value(s) of the proposed
      momentum(s) of the state (parts).
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    shard_axis_names: A structure of string names indicating how
      members of the state are sharded.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').

  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    def compute_sum_sq(v, shard_axes):
      sum_sq = tf.reduce_sum(v ** 2., axis=ps.range(
          independent_chain_ndims, ps.rank(v)))
      if shard_axes is not None:
        sum_sq = distribute_lib.psum(sum_sq, shard_axes)
      return sum_sq
    shard_axis_names = (shard_axis_names or ([None] * len(current_momentums)))
    current_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(current_momentums, shard_axis_names)])
    proposed_kinetic = tf.add_n([
        compute_sum_sq(v, axes) for v, axes
        in zip(proposed_momentums, shard_axis_names)])
    return 0.5 * mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])
