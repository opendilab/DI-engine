Impala
^^^^^^^

Overview
---------
IMPALA, or the Importance Weighted Actor Learner Architecture, is an off-policy actor-critic framework that
decouples acting from learning and learns from experience trajectories using V-trace. This method is first
introduced in `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures <https://arxiv.org/abs/1802.01561>`_.


Quick Facts
-------------
1. Impala  is a **model-free** and **off-policy** RL algorithm.

2. Impala can support both **discrete** action spaces and **continuous** action spaces.

3. Impala is a actor-critic RL algorithm, which optimizes actor network and critic network, respectively.

4. Impala decouples acting from learning. Collectors in impala will not compute value or advantage.


Key Equations
---------------------------
Loss used in Impala is similar to that in PPO, A2C and other actor-critic model. All of them comes from policy_loss,\
value_loss and entropy_loss, with respect to some carefully chosen weights.

.. math::
    :label: math-single

    Loss_total = Loss_policy + w_value * Loss_value + w_entropy * Loss_entropy

where  w_value, w_entropy are loss weights for value and entropy.

- NOTATION AND CONVENTIONS:

:math:`\pi_{\phi}`: current training policy parameterized by :math:`\phi`.

:math:`V_\theta`: value function parameterized by :math:`\theta`.

:math:`\mu`: older policy which generates trajectories in replay buffer.


At the training time :math:`t`, given transition :math:`(x_t, a_t, x_{t+1}, r_t)`, the value function :math:`V_\theta`
is learned through an :math:`L_2` loss between the current value and a V-trace target value. The n-step V-trace target
at time s is defined as follows:

.. math::
    :label: math-single

    v_s  \stackrel{def}{=} V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \big(\prod_{i=s}^{t-1} c_i\big)\delta_t V

where :math:`\delta_t V \stackrel{def}{=}  \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))` is a temporal difference for :math:`V`.

:math:`\rho_t \stackrel{def}{=} \min\big(\bar{\rho}, \frac{\pi(a_t \vert x_t)}{\mu(a_t \vert x_t)}\big)` and :math:`c_i \stackrel{def}{=}
\min\big(\bar{c}, \frac{\pi(a_i \vert s_i)}{\mu(a_i \vert s_i)}\big)` are truncated importance sampling (IS) weights,
where :math:`\bar{\rho}` and :math:`\bar{c}` are two truncation constants with :math:`\bar{\rho} \geq \bar{c}`.

The product of :math:`c_s, \dots, c_{t-1}` measures how much a temporal difference :math:`\delta_t V` observed at time
:math:`t` impacts the update of the value function at a previous time :math:`s` . In the on-policy case, we have :math:`\rho_t=1`
and :math:`c_i=1` (assuming :math:`\bar{c} \geq 1)` and therefore the V-trace target becomes on-policy n-step Bellman
target.


:math:`\bar{\rho}` impacts the fixed-point of the value function we converge to,and :math:`\bar{c}` impacts the speed
of convergence. When :math:`\bar{\rho} =\infty` (untruncated), v-trace value function will converge to the value
function of the target policy :math:`V_\pi`; when :math:`\bar{\rho}` is close to 0, we evaluate the value function
of the behavior policy :math:`V_\mu`; when in-between, we evaluate a policy between :math:`\pi` and :math:`\mu`.

Therefore, loss functions are

.. math::
    :label: math-mult

    Loss_value &= (v_s - V_\theta(x_s))^2 \\
    Loss_policy &= -\rho_s \log \pi_\phi(a_s \vert x_s)  \big(r_s + \gamma v_{s+1} - V_\theta(x_s)\big) \\
    Loss_entropy &= -H(\pi_\phi) = \sum_a \pi_\phi(a\vert x_s)\log \pi_\phi(a\vert x_s)

where :math:`H(\pi_{\phi})`, entropy of policy :math:`\phi`, is an bonus to encourage exploration.

Value function parameter is updated in the direction of:

.. math::
    :label: math-single

    \Delta\theta = w_value (v_s - V_\theta(x_s))\nabla_\theta V_\theta(x_s)

Policy parameter :math:`\phi` is updated through policy gradient,

.. math::
    :label: math-mult

    \Delta \phi &= \rho_s \nabla_\phi \log \pi_\phi(a_s \vert x_s) \big(r_s + \gamma v_{s+1}- V_\theta(x_s)\big)\\
                &- w_entropy \nabla_\phi \sum_a \pi_\phi(a\vert x_s)\log \pi_\phi(a\vert x_s)

where :math:`r_s + \gamma v_{s+1}` is the v-trace advantage, which is estimated Q value subtracted by a state-dependent baseline :math:`V_\theta(x_s)`.



Key Graphs
---------------
The following graph describes the process in IMPALA original paper. However, our implication is a little different from
that in original paper.

.. image:: images/IMPALA.png

For single learner, they use multiple actors/collectors to generate training data. While in our
setting, we use one collector which has multiple environments to increase data diversity.

For multiple learner, in original paper, different learners will have different actors with them. In other word, they
will have different ReplayBuffer. While in our setting, all of the learners, will share the same ReplayBuffer, and will
synchronize after each iteration.



Implementations
----------------
The default config is defined as follows:

.. autoclass:: nervex.policy.impala.IMPALAPolicy

Usually, we hope to compute everything as a batch to improve efficiency. Especially, when computing vtrace, we
need all training sample (sequences of training data) have the same length. This is done in ``policy._get_train_sample``.
Once we execute this function in collector, the length of samples will equal to unroll-len in config. For details, please
refer to doc of ``adder``.

.. code:: python
    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._adder.get_train_sample(data)

    def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
            If ``self._unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
            Otherwise, ``data`` will be split according to ``self._unroll_len``, process residual part according to
            ``self._last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
        Returns:
            - data (:obj:`List[Dict[str, Any]]`): transitions list processed after unrolling
        """
        if self._unroll_len == 1:
            return data
        else:
            # cut data into pieces whose length is unroll_len
            split_data, residual = list_split(data, step=self._unroll_len)

            def null_padding():
                template = copy.deepcopy(residual[0])
                template['done'] = True
                template['reward'] = torch.zeros_like(template['reward'])
                if 'value_gamma' in template:
                    template['value_gamma'] = 0.
                null_data = [self._get_null_transition(template) for _ in range(miss_num)]
                return null_data

            if residual is not None:
                miss_num = self._unroll_len - len(residual)
                if self._last_fn_type == 'drop':
                    # drop the residual part
                    pass
                elif self._last_fn_type == 'last':
                    if len(split_data) > 0:
                        # copy last datas from split_data's last element, and insert in front of residual
                        last_data = copy.deepcopy(split_data[-1][-miss_num:])
                        split_data.append(last_data + residual)
                    else:
                        # get null transitions using ``null_padding``, and insert behind residual
                        null_data = null_padding()
                        split_data.append(residual + null_data)
                elif self._last_fn_type == 'null_padding':
                    # same to the case of 'last' type and split_data is empty
                    null_data = null_padding()
                    split_data.append(residual + null_data)
            # collate unroll_len dicts according to keys
            if len(split_data) > 0:
                split_data = [lists_to_dicts(d, recursive=True) for d in split_data]
            return split_data

.. note::
    In ``adder.get_train_sample``, we introduce three ways to cut trajectory data into same-length pieces (length equal
    to ``unroll_len``).

    The first one is ``drop``, this means after splitting trajectory data into small pieces, we simply throw away those
    with length smaller than ``unroll_len``. This method is kind of naive and usually is not a good choice. Since in
    Reinforcement Learning, the last few data in an episode is usually very important, we can't just throw away them.

    The second method is ``last``, which means if the total length trajectory is smaller than ``unrollen_len``,
    we will use zero padding. Else, we will use data from previous piece to pad residual piece. This method is set as
    default and recommended.

    The last method ``null_padding`` is just zero padding, which is not vert efficient since many data will be ``null``.


Now, we introduce the computation of vtrace-value.
First, we use the following functions to compute importance_weights.

.. code:: python
    def compute_importance_weights(target_output, behaviour_output, action, requires_grad=False):
        """
        Shapes:
            - target_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where T is timestep, B is batch size and\
                N is action dim
            - behaviour_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(T, B)`
            - rhos (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """

        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        assert isinstance(action, torch.Tensor)
        device = action.device

        with grad_context:
            dist_target = torch.distributions.Categorical(logits=target_output)
            dist_behaviour = torch.distributions.Categorical(logits=behaviour_output)
            rhos = dist_target.log_prob(action) - dist_behaviour.log_prob(action)
            rhos = torch.exp(rhos)
            return rhos


After that, we clip importance weights based on constant :math:`\rho` and :math:`c` to get clipped_rhos, clipped_cs.
Then we can compute vtrace value according to the following function. Notice, here bootstrap_values are just
value function :math:`V(x_s)` in vtrace definition.

.. code:: python
    def vtrace_nstep_return(clipped_rhos, clipped_cs, reward, bootstrap_values, gamma=0.99, lambda_=0.95):
        """
        Shapes:
            - clipped_rhos (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep, B is batch size
            - clipped_cs (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - reward: (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - bootstrap_values (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - vtrace_return (:obj:`torch.FloatTensor`):  :math:`(T, B)`
        """
        deltas = clipped_rhos * (reward + gamma * bootstrap_values[1:] - bootstrap_values[:-1])
        factor = gamma * lambda_
        result = bootstrap_values[:-1].clone()
        vtrace_item = 0.
        for t in reversed(range(reward.size()[0])):
            vtrace_item = deltas[t] + factor * clipped_cs[t] * vtrace_item
            result[t] += vtrace_item
        return result


.. note::
    1. Bootstrap_values in this part need to have size (T+1,B),where T is timestep, B is batch size. The reason is that
    we need a sequence of training data with same-length vtrace value (this length is just the unroll_len in config).
    And in order to compute the last vtrace value in the sequence, we need at least one more target value. This is
    done using the next_obs of the last transition in training data sequence.

    2. Here we introduce a parameter ``lambda_``, following the implementation in AlphaStar. The parameter, between 0
    and 1,can give a subtle control on vtrace off-policy correction. Usually, we will choose this parameter close to 1.

Once we get vtrace value, or ``vtrace_nstep_return``, the computation of loss functions are straightforward. The whole
process is as follows.

.. code:: python
    def vtrace_advantage(clipped_pg_rhos, reward, return_, bootstrap_values, gamma):
        """
        Shapes:
            - clipped_pg_rhos (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep, B is batch size
            - reward: (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - return_ (:obj:`torch.FloatTensor`):  :math:`(T, B)`
            - bootstrap_values (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - vtrace_advantage (:obj:`torch.FloatTensor`):  :math:`(T, B)`
        """
        return clipped_pg_rhos * (reward + gamma * return_ - bootstrap_values)

    def vtrace_error(
            data: namedtuple,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            rho_clip_ratio: float = 1.0,
            c_clip_ratio: float = 1.0,
            rho_pg_clip_ratio: float = 1.0):
        """
        Shapes:
            - target_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where T is timestep, B is batch size and\
                N is action dim
            - behaviour_output (:obj:`torch.FloatTensor`): :math:`(T, B, N)`
            - action (:obj:`torch.LongTensor`): :math:`(T, B)`
            - value (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - reward (:obj:`torch.LongTensor`): :math:`(T, B)`
            - weight (:obj:`torch.LongTensor`): :math:`(T, B)`
        """

        target_output, behaviour_output, action, value, reward, weight = data
        with torch.no_grad():
            IS = compute_importance_weights(target_output, behaviour_output, action)
            rhos = torch.clamp(IS, max=rho_clip_ratio)
            cs = torch.clamp(IS, max=c_clip_ratio)
            return_ = vtrace_nstep_return(rhos, cs, reward, value, gamma, lambda_)
            pg_rhos = torch.clamp(IS, max=rho_pg_clip_ratio)
            return_t_plus_1 = torch.cat([return_[1:], value[-1:]], 0)
            adv = vtrace_advantage(pg_rhos, reward, return_t_plus_1, value[:-1], gamma)

        if weight is None:
            weight = torch.ones_like(reward)
        dist_target = torch.distributions.Categorical(logits=target_output)
        pg_loss = -(dist_target.log_prob(action) * adv * weight).mean()
        value_loss = (F.mse_loss(value[:-1], return_, reduction='none') * weight).mean()
        entropy_loss = (dist_target.entropy() * weight).mean()
        return vtrace_loss(pg_loss, value_loss, entropy_loss)

.. note::
    1. The shape of value in input data should be (T+1, B), the reason is same as above Note.
    2. Here we introduce a parameter ``rho_pg_clip_ratio``, following the implementation in AlphaStar. This parameter,
    can give a subtle control on vtrace advantage. Usually, we will choose this parameter just same as rho_clip_ratio.

The network interface IMPALA used is defined as follows:

    * TODO(similar to that in A2C,PPO)

The Benchmark result of IMPALA implemented in nerveX is shown in `Benchmark <../feature/algorithm_overview.html>`_



