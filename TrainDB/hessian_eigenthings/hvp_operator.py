from hessian_eigenthings.power_iter import Operator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
import torch

class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian forrad_vec
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
        self,
        model,
        dataloader,
        criterion,
        grad_vec,
        weird=False,
        use_gpu=False,
        full_dataset=True,
        max_samples=256,
        opt = False
    ):
        size = int(sum(p.numel() for p in model.parameters()))
        #print(size)
        #size = int(sum(p.numel() for p in model.parameters() if len(p.size()) > 1 ))
        #print(size)
        super(HVPOperator, self).__init__(size,opt)
        self.grad_vec = grad_vec
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.full_dataset = full_dataset
        self.max_samples = max_samples
        self.weird = weird
        #self.opt = opt

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.opt:
            return self._apply_batch2(vec)
        else:

            if self.full_dataset:
                return self._apply_full(vec)
            else:
                return self._apply_batch(vec)


    def _apply_batch(self, vec):
        # compute original gradient, tracking computation graph
        self.zero_grad()
        grad_vec = self.prepare_grad()
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
        )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        return hessian_vec_prod

    def _apply_batch2(self, vec):
        # compute original gradient, tracking computation graph
        self.zero_grad()
        w = [p for p in self.model.parameters()]
        #print(torch.norm(self.grad_vec))
        grad_grad = torch.autograd.grad(
            self.grad_vec, w, grad_outputs=vec, only_inputs=True,retain_graph=True
            #, allow_unused=True
        )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        #print(hessian_vec_prod,vec)
        return hessian_vec_prod

    def _apply_full(self, vec):
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod
    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = max(1, len(all_inputs) // self.max_samples)

        grad_vec = None

        input_chunks = all_inputs.chunk(num_chunks)
        target_chunks = all_targets.chunk(num_chunks)
        for input, target in zip(input_chunks, target_chunks):
            if self.use_gpu:
                input = input.cuda()
                target = target.cuda()

            if self.weird:
                input = input.view(input.shape[0], -1)
            output = self.model(input)
            loss = self.criterion(output, target)
            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec


    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()


def compute_hessian_eigenthings(
    model,
    dataloader,
    loss,
    grad_vec,
    weird = False,
    num_eigenthings=10,
    full_dataset=True,
    mode="power_iter",
    use_gpu=False,
    max_samples=512,
    opt=False,
    **kwargs
):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_eigenthings : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
    mode : str ['power_iter', 'lanczos']
        which backend to use to compute the top eigenvalues.
    use_gpu:
        if true, attempt to use cuda for all lin alg computatoins
    max_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(
        model,
        dataloader,
        loss,
        grad_vec,
        weird = weird,
        use_gpu=use_gpu,
        full_dataset=full_dataset,
        max_samples=max_samples,
        opt=opt
    )
    eigenvals, eigenvecs = None, None
    if mode == "power_iter":
        eigenvals, eigenvecs = deflated_power_iteration(
            hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
        )
    elif mode == "lanczos":
        eigenvals, eigenvecs = lanczos(
            hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
        )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return eigenvals, eigenvecs
