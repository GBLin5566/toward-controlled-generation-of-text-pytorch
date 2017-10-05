from torch import cuda

def check_cuda(torch_var, use_cuda=False):
    if use_cuda and cuda.is_available():
        return torch_var.cuda()
    else:
        return torch_var
