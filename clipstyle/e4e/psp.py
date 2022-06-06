import torch
from torch import nn
from .encoders import psp_encoders

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        # Load weights if needed
        self.load_weights()

    def load_weights(self):
        print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.ckpt))
        ckpt = torch.load(self.opts.ckpt, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt['state_dict'], 'encoder'), strict=True)
        self.__load_latent_avg(ckpt)

    def forward(self, x, latent_mask=None, input_code=False, inject_latent=None, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        return codes

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            latent_avg = ckpt['latent_avg']
            if repeat is not None:
                latent_avg = latent_avg.repeat(repeat, 1)
            self.register_buffer('latent_avg', latent_avg)
        
