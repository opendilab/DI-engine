import torch
import torch.nn.functional as F
from ding.model import ConvEncoder, FCEncoder


class EMAverager:
    def __init__(self):
        self._update_rate = 0.01
        self._average = torch.Tensor([0])
        self._mass = 0
    
    def update(self, new_val: torch.Tensor):
        new_val = new_val.detach()
        self._average += (new_val - self._average) * self._update_rate
        self._mass += (1 - self._mass) * self._update_rate
    
    def get(self):
        return self._average.detach() / self._mass


class MIEstimator:
    def __init__(self, obs_shape):
        self._x_dim = 1
        self._y_dim = 1
        self._type = "DV"
        self._model = FCEncoder(
            obs_shape=obs_shape,
            hidden_size_list=[obs_shape * 2, obs_shape, 1]   # don't use too deep networks!
        )
        self._sampler = self._shift_sampler
        self._mean_averager = EMAverager()
    

    def _shift_sampler(self, y, shift=1):
        y = torch.concat([y[-shift:, ...], y[0:-shift, ...]], axis=0).detach()
        return y
    
    def _shuffle_sampler(self, y):
        idx = torch.randperm(y.shape[0])
        y = torch.Tensor(y[idx]).detach()
        return y

    def forward(self, inputs):
        """Perform training on one batch of inputs.
        Args:
            inputs (tuple(Tensor, Tensor)): tuple of x and y
            state: not used
        Returns:
            AlgorithmStep
                outputs (Tensor): shape=[batch_size], its mean is the estimated
                    MI
                state: not used
                info (LossInfo): info.loss is the loss
        """
        x, y = inputs

        #TODO: flatten x, y
        y_neg = self._sampler(y)

        joint = torch.concat([x, y], axis=1)
        marginal = torch.concat([x, y_neg], axis=1)

        log_ratio = self._model.forward(joint)
        t1 = torch.exp(self._model.forward(marginal))


        ratio = torch.exp(torch.clip(t1, max=20))
        with torch.no_grad():
            mean = torch.mean(ratio)
            if self._mean_averager:
                self._mean_averager.update(mean)
                unbiased_mean = self._mean_averager.get()
            else:
                unbiased_mean = mean
            # estimated MI = reduce_mean(mi)
            # ratio/mean-1 does not contribute to the final estimated MI, since
            # mean(ratio/mean-1) = 0. We add it so that we can have an estimation
            # of the variance of the MI estimator
            mi = log_ratio - (torch.log(mean) + ratio / mean - 1)

        loss = torch.mean(ratio) / unbiased_mean - torch.mean(log_ratio)
        
        # loss = torch.mean(ratio / unbiased_mean - joint)

        #TODO: unflatten

        return loss
    

    def eval(self, inputs):
        """Return estimated pointwise mutual information.
        The pointwise mutual information is defined as:
            log P(x|y)/P(x) = log P(y|x)/P(y)
        Args:
            x (tf.Tensor): x
            y (tf.Tensor): y
        Returns:
            tf.Tensor: pointwise mutual information between x and y
        """
        x, y = inputs
        joint = torch.concat([x, y], axis=1)
        log_ratio = self._model(joint)
        if self._type == 'DV':
            log_ratio -= torch.log(self._mean_averager.get())
        return torch.mean(log_ratio)


class InfoNCE:
    def __init__(self, obs_shape):
        self._model = FCEncoder(
            obs_shape=obs_shape,
            hidden_size_list=[32, 16, 1]   # don't use too deep networks!
        )

    def forward(l: torch.Tensor, m: torch.Tensor):
        '''Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.

        Note that vectors should be sent as 1x1.

        Args:
            l: Local feature map.
            m: Multiple globals feature map.

        Returns:
            torch.Tensor: Loss.

        '''
        N, units, n_locals = l.size()
        _, _ , n_multis = m.size()

        # First we make the input tensors the right shape.
        l_p = l.permute(0, 2, 1)
        m_p = m.permute(0, 2, 1)

        l_n = l_p.reshape(-1, units)
        m_n = m_p.reshape(-1, units)

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
        u_p = torch.matmul(l_p, m).unsqueeze(2)
        u_n = torch.mm(m_n, l_n.t())
        u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
        u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0].mean()

        return loss