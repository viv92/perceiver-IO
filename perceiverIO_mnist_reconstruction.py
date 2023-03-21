### Program implementing PerceiverIO and training for reconstructing MNIST images

## Features:
# 1. Key idea is to obtain a latent code that acts as a low dimensional bottleneck representation for a high-dimensional input vector / embedding and then decode it to the desired output dimension
# 2. Architecture: a perceiver block consists of a cross-attention layer followed by a latent transformer. The latent transformer's architecture is based on the gpt2 architecture. But no causal masks are used. The entire perceiver module just applies the perceiver block in a recurrent fashion (like an RNN cell). This recurrence enables weight sharing (except for the first cross-attention layer). Also, the decoupling of input dimension and the latent dimension allows for a very deep perceiver module (depth / number of layers = number of recurrence calls to perceiver block). Finally, the latent code is decoded into a desired output dimension by using cross-attention with a learnt output vector as the query and the latent code as the key and value.
# 3. Note that perceiverIO can support arbitrary input and output dimensions by basically flattening all but last dimensions into a 1-d vector. The last dimension is treated as the channel dimension which is projected into the d_model dimension used for attention modules. 

## Todos / Questions:
# 1. Positional embeddings: added or concatenated? The paper mentions concatenation is better than addition, but I don't get why. 
# 2. Positional embeddings: learnt or fixed fourier? The paper results show fixed fourier positional embeddings to perform better for imagenet task.
# 3. How is the latent bottleneck sustained during cross-attention? Let the latent code be z and the input be x. Let z.shape = (N, D) and x.shape = (M, C). Now to apply cross_attention, we need to project z and x into embeddings such that z_emb.shape = (N, d_model) and x_emb.shape = (M, d_model). But now z_emb is no longer a latent bottleneck as z_emb.shape[-1] == x_emb.shape[-1]. Probably the bottleneck is only by virtue of N << M.
# 4. Why do we need positional embeddings for z, when it's a randomly initialized tensor anyway?
# 5. From reference implementations, it seems like even the cross-attention block is wrapped by sublayer connections as in a typical transformer. So each cross-attention block is actually like a transformer layer: dropout(gelu(ff(norm( dropout(xattn(norm(z), x)) + z ))) + z
# 6. In perceiverIO, does the latent processing involve only the latent transformer, or both the xattn_encoder and the latent transformer (similar to original perceiver) 


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from time import time 
import cv2 

from utils_transformer import * 

# class implementing perceiver cross-attention block for encoding 
class Perceiver_xattn_encoder(nn.Module):
    def __init__(self, xattn, feed_forward, dim, dropout):
        super().__init__()
        self.xattn = xattn
        self.feed_forward = feed_forward
        self.ln1_q = nn.LayerNorm(dim)
        self.ln1_kv = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, z, mask_padding):
        x = self.ln1_kv(x)
        z1 = self.ln1_q(z)
        z1 = self.xattn(x, z1, x, mask_padding=mask_padding) # xattn: (key=x, query=z, value=x)
        z = self.dropout(z1) + z 
        z1 = self.ln2(z)
        z1 = self.feed_forward(z1)
        z = self.dropout(z1) + z 
        return z
    
# class implementing perceiver cross-attention block for decoding 
class Perceiver_xattn_decoder(nn.Module):
    def __init__(self, xattn, feed_forward, dim, dropout):
        super().__init__()
        self.xattn = xattn
        self.feed_forward = feed_forward
        self.ln1_q = nn.LayerNorm(dim)
        self.ln1_kv = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, o, z, mask_padding):
        z = self.ln1_kv(z)
        o1 = self.ln1_q(o)
        o1 = self.xattn(z, o1, z, mask_padding=mask_padding) # xattn: (key=x, query=o, value=x)
        o = self.dropout(o1) # note no residual connection here for decoder (as mentioned in perceiverIO paper: appendix E1) 
        o1 = self.ln2(o)
        o1 = self.feed_forward(o1)
        o = self.dropout(o1) + o 
        return o
    
# class implementing perceiver IO
class PerceiverIO(nn.Module):
    def __init__(self, xattn_encoder, xattn_decoder, latent_transformer, depth, x_seqlen, x_dim, z_seqlen, o_seqlen, o_dim, d_model, device):
        super().__init__()
        self.xattn_encoder_unique = xattn_encoder # first cross attention block is unique (weights not shared with other cross-attention blocks)
        self.xattn_encoder_shared = deepcopy(xattn_encoder) # rest of the cross-attention blocks with shared weights 
        self.latent_transformer = latent_transformer # all latent transformer blocks share weights
        self.xattn_decoder = xattn_decoder # cross-attention block for decoding
        self.depth = depth 
        self.z = nn.Parameter(torch.rand(z_seqlen, d_model)) # randomly initialized input latent code with z_dim = d_model 
        self.x_emb = nn.Linear(x_dim, d_model, bias=False)
        self.o = nn.Parameter(torch.rand(o_seqlen, d_model)) # randomly initialized output latent code with o_dim = d_model 
        self.z_pos_emb = nn.Parameter(torch.rand(z_seqlen, d_model)) # learnt positional embeddings
        self.x_pos_emb = nn.Parameter(torch.rand(x_seqlen, d_model)) # learnt positional embeddings
        self.o_pos_emb = nn.Parameter(torch.rand(o_seqlen, d_model)) # learnt positional embeddings
        self.ln_out = nn.LayerNorm(d_model) # layernorm on final output
        self.proj_head = nn.Linear(d_model, o_dim) # final projection to o_dim (desired channel dimension of output)
    def forward(self, x): # x.shape: [b, M, C]
        batch_size = x.shape[0]
        x = self.x_emb(x) + self.x_pos_emb # x.shape: [b, M, d_model]
        z = self.z + self.z_pos_emb # z.shape: [N, d_model]
        o = self.o + self.o_pos_emb # o.shape: [O, d_model]
        z = z.unsqueeze(0).expand(batch_size, -1, -1) # z.shape: [b, N, d_model]
        o = o.unsqueeze(0).expand(batch_size, -1, -1) # o.shape: [b, O, d_model]
        
        # forward prop through first cross-attention encoder block 
        z = self.xattn_encoder_unique(x, z, mask_padding=None) # key = x, query = z, value = x
        # forward prop through first latent transformer block 
        z = self.latent_transformer(z, mask_padding=None, mask_causal=None)
        # forward prop through rest of the blocks
        for _ in range(self.depth-1):
            z = self.xattn_encoder_shared(x, z, mask_padding=None) # key = x, query = z, value = x
            z = self.latent_transformer(z, mask_padding=None, mask_causal=None)
        # decode the latent code to output vector
        o = self.xattn_decoder(o, z, mask_padding=None) # key = z, query = o, value = z
        # apply final layernorm 
        o = self.ln_out(o) # o.shape: [b, o_seqlen, d_model]
        # project channel dimension of the output to the required dimension
        o = self.proj_head(o) # o.shape: [b, O = o_seqlen, E = o_dim]
        return o
    
# function to instantiate perceiver
def init_perceiver(depth, x_seqlen, x_dim, z_seqlen, z_dim, o_seqlen, o_dim, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    attn_singleHead = MultiHeadAttention(1, d_model, d_model, d_model, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    latent_transformer = Encoder(encoder_layer, n_layers, d_model) # latent transformer is just the transformerr encoder = stacked encoder layers
    xattn_encoder = Perceiver_xattn_encoder(deepcopy(attn_singleHead), deepcopy(ff), d_model, dropout) # perceiver cross_attn block for encoding
    xattn_decoder = Perceiver_xattn_decoder(deepcopy(attn_singleHead), deepcopy(ff), d_model, dropout) # perceiver cross_attn block for decoding
    perceiver = PerceiverIO(xattn_encoder, xattn_decoder, latent_transformer, depth, x_seqlen, x_dim, z_seqlen, o_seqlen, o_dim, d_model, device)
    # initialize params - Xavier initialization
    for p in perceiver.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return perceiver

# function to get test loss 
def get_test_loss(perceiver, testloader, criterion):
    print('Testing...')
    pbar = tqdm(total=len(testloader))
    with torch.no_grad():
        test_loss = 0
        for i, data in enumerate(testloader):
            imgs, labels = data[0].to(device),data[1].to(device)
            # flatten imgs to a sequence of pixels 
            imgs = imgs.flatten(start_dim=2, end_dim=3) # imgs.shape: [b, c, h*w]
            imgs = imgs.permute(0, 2, 1) # imgs.shape: [b, h*w, c]
            imgs_reconstructed = perceiver(imgs)
            loss = criterion(imgs_reconstructed, imgs)
            test_loss += loss
            pbar.update(1)
        test_loss = test_loss / len(testloader)
    pbar.close()
    return test_loss.item()

# function to save a test img and its reconstructed img 
def save_test_img_reconstructed(test_img, perceiver, save_path):
    recons_img = perceiver(test_img.unsqueeze(0))
    recons_img = recons_img.squeeze(0).reshape(img_size, img_size, 1)
    test_img = test_img.reshape(img_size, img_size, 1)
    concat_img = torch.cat([test_img, recons_img], dim=1)
    concat_img = concat_img.detach().cpu().numpy()
    concat_img = np.uint8( concat_img * 255 )
    cv2.imwrite(save_path, concat_img)

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


# main - training Perceiver on MNIST
if __name__ == '__main__':

    # hyperparams for perceiver 
    d_model = 128 # 768 based on d_model for gpt2
    img_size = 28 # as per MNIST dataset 
    img_channels = 1 # for MNIST
    z_dim = d_model # D in paper fig.2
    z_seqlen = 16 # N in paper fig.2 - so the compression is from [28*28, d_model] -> [z_seqlen, d_model]
    x_dim = img_channels # C in paper fig.2
    x_seqlen = img_size * img_size # M in paper fig.2
    o_seqlen = x_seqlen # O in paper fig.2
    o_dim = img_channels # E in paper fig.2
    depth = 8 # depth of perceiver model = number of recurrence calls to perceiver block 
    
    # hyperparams for latent transformer
    n_heads = 4
    assert (d_model % n_heads) == 0
    d_k = d_model // n_heads 
    d_v = d_k
    n_layers = 8 
    d_ff = d_model * 4
    dropout = 0.1
    batch_size = 128
    lr = 3e-4
    num_epochs = 5
    num_evals_per_epoch = 1
    random_seed = 10

    checkpoint_path = 'ckpts/perceiverIO_lnfix_mnist_reconstruction.pt' # path to a save and load checkpoint of the trained model
    resume_training_from_ckpt = True    

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init model
    perceiver = init_perceiver(depth, x_seqlen, x_dim, z_seqlen, z_dim, o_seqlen, o_dim, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # init optimizer
    optimizer = torch.optim.AdamW(params=perceiver.parameters(), lr=lr)

    if resume_training_from_ckpt:
        perceiver, optimizer = load_ckpt(checkpoint_path, perceiver, optimizer=optimizer, device=device, mode='train')

    # loss function
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    # load MNIST dataset
    resize_shape = (img_size, img_size)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
        torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    ])
    trainset = torchvision.datasets.MNIST(root='./dataset_mnist', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    testset = torchvision.datasets.MNIST(root='./dataset_mnist', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # set up tqdm progress bar 
    pbar = tqdm(total=len(trainloader) * num_epochs)

    # train
    results_train_loss = []
    results_test_loss = []
    start_time = time()
    for epoch in range(num_epochs):
        running_loss = 0
        epoch_loss = 0
        for i, data in enumerate(trainloader, 0):
            imgs, labels = data[0].to(device), data[1].to(device) # imgs.shape: [b, c, h, w]
            # flatten imgs to a sequence of pixels 
            imgs = imgs.flatten(start_dim=2, end_dim=3) # imgs.shape: [b, c, h*w]
            imgs = imgs.permute(0, 2, 1) # imgs.shape: [b, h*w, c]

            optimizer.zero_grad()
            imgs_reconstructed = perceiver(imgs)
            loss = loss_fn(imgs_reconstructed, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i+1) % (len(trainloader) // num_evals_per_epoch) == 0:
                test_loss = get_test_loss(perceiver, testloader, loss_fn)
                # print('epoch:{} \t i:{} \t train_loss:{} \t test_loss:{} \t test_accuracy:{}'.format(epoch, i, running_loss/2000, test_loss/2000, test_accuracy))
                results_train_loss.append(running_loss / (len(trainloader) // num_evals_per_epoch))
                results_test_loss.append(test_loss)
                running_loss = 0

            pbar.update(1)

        print('train_epoch_loss: ', epoch_loss / len(trainloader))
        print('test_loss: ', results_test_loss[-1])

        # save model checkpoint
        save_ckpt(device, checkpoint_path, perceiver, optimizer)

        # get a test img and its reconstructed img
        test_img = imgs[0] # shape: []
        save_path = 'plots/test_img_epoch=' + str(epoch) + '.png'
        save_test_img_reconstructed(test_img, perceiver, save_path)

    pbar.close()
    end_time = time()
    time_taken = end_time - start_time

    hyperparam_dict = {}
    hyperparam_dict['z_dim'] = z_dim
    hyperparam_dict['z_seqlen'] = z_seqlen
    hyperparam_dict['depth'] = depth
    hyperparam_dict['d_model'] = d_model
    hyperparam_dict['n_heads'] = n_heads
    hyperparam_dict['n_layers'] = n_layers
    hyperparam_dict['d_ff'] = d_ff 
    hyperparam_dict['dropout'] = dropout 
    hyperparam_dict['batch_size'] = batch_size
    hyperparam_dict['lr'] = lr 
    hyperparam_dict['epochs'] = num_epochs 
    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += ';' + k + ':' + str(v)

    # plot results
    x = np.arange(len(results_test_loss))
    plt.plot(x, results_train_loss, label='train_loss')
    plt.plot(x, results_test_loss, label='test_loss')
    plt.legend()
    plt.title('final_test_loss: ' + str(results_test_loss[-1]))
    plt.savefig('plots/perceiverIO_lnfix_mnist_reconstruction_DataAug' + hyperparam_str + '.png')