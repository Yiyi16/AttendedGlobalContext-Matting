import torch
import torch.nn as nn

from   utils import CONFIG
from   networks import encoders, decoders
import operator


class Generator(nn.Module):
    def __init__(self, encoder, encoder2, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap, trans1, mask1, trans2, mask2):
        inp = torch.cat((image, trimap), dim=1)
        inp2 = torch.cat((trans1, mask1), dim=1)
        inp3 = torch.cat((trans2, mask2), dim=1)
        embedding, mid_fea = self.encoder(inp, inp2, inp3)
        #embedding2, mid_fea2 = self.encoder2(torch.cat((trans1, mask1), dim=1))
        #embedding = torch.add(embedding, 0.8*embedding2)
        #mid_fea['shortcut'] = tuple(map(operator.add, mid_fea['shortcut'], 0.8*mid_fea2['shortcut']))
        #mid_fea['unknown'] = torch.add(mid_fea['unknown'], 0.8*mid_fea2['unknown'])
        alpha, bg, info_dict = self.decoder(embedding, mid_fea)
        return alpha, bg, info_dict


def get_generator(encoder, encoder2, decoder):
    generator = Generator(encoder=encoder, encoder2=encoder2, decoder=decoder)
    return generator


if __name__=="__main__":
    import time
    generator = get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder).cuda().train()
    batch_size = 12
    # generator.eval()
    n_eval = 10
    # pre run the model
    # with torch.no_grad():
    #     for i in range(2):
    #         x = torch.rand(batch_size, 3, 512, 512, device=device)
    #         y = torch.rand(batch_size, 3, 512, 512, device=device)
    #         z = generator(x,y)
    # test without GPU IO

    # x = torch.zeros(batch_size, 3, 512, 512, device=device)
    # y = torch.zeros(batch_size, 1, 512, 512, device=device)
    x = torch.randn(batch_size, 3, 512, 512)
    y = torch.randn(batch_size, 3, 512, 512)
    t = time.time()
    # with torch.no_grad():
    for i in range(n_eval):
        a = generator(x.cuda(),y.cuda())
    torch.cuda.synchronize()
    #print(generator.__class__.__name__, 'With IO  \t', f'{(time.time() - t)/n_eval/batch_size:.5f} s')
    #print(generator.__class__.__name__, 'FPS \t\t', f'{1/((time.time() - t)/n_eval/batch_size):.5f} s')
    for n, p in generator.named_parameters():
        print(n)
