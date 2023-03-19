### Program implementing Perceiver and training for MNIST classification task

## Features:
# 1. Key idea is to obtain a latent code that acts as a low dimensional bottleneck representation for a high-dimensional input vector / embedding.
# 2. Architecture: a perceiver block consists of a cross-attention layer followed by a latent transformer. The latent transformer's architecture is based on the gpt2 architecture. But no causal masks are used. The entire perceiver module just applies the perceiver block in a recurrent fashion (like an RNN cell). This recurrence enables weight sharing. Also, the decoupling of input dimension and the latent dimension allows for a very deep perceiver module (depth / number of layers = number of recurrence calls to perceiver block).

## Todos / Questions:
# 1. Positional embeddings: added or concatenated? The paper mentions concatenation is better than addition, but I don't get why. 
# 2. Positional embeddings: learnt or fixed fourier? The paper results show fixed fourier positional embeddings to perform better for imagenet task.
# 3. How is the latent bottleneck sustained during cross-attention? Let the latent code be z and the input be x. Let z.shape = (N, D) and x.shape = (M, C). Now to apply cross_attention, we need to project z and x into embeddings such that z_emb.shape = (N, d_model) and x_emb.shape = (M, d_model). But now z_emb is no longer a latent bottleneck as z_emb.shape[-1] == x_emb.shape[-1]. Probably the bottleneck is only by virtue of N << M.
# 4. Why do we need positional embeddings for z, when it's a randomly initialized tensor anyway?
# 5. From reference implementations, it seems like even the cross-attention block is wrapped by sublayer connections as in a typical transformer. So each cross-attention block is actually like a transformer layer: dropout(gelu(ff(norm( dropout(xattn(norm(z), x)) + z ))) + z
# 6. Final logits are obtained by averaging the latents


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from time import time 

from utils_transformer import * 

# class implementing perceiver cross-attention block 
class Perceiver_xattn(nn.Module):
    def __init__(self, xattn, feed_forward, dim, dropout):
        super().__init__()
        self.xattn = xattn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # first for xattn block, second for feed_forward block
    def forward(self, x, z, mask_padding):
        z = self.sublayers[0](z, lambda z: self.xattn(x, z, x, mask_padding=mask_padding)) # xattn: (key=x, query=z, value=x)
        z = self.sublayers[1](z, self.feed_forward)
        return z
    
# class implementing perceiver 
class Perceiver(nn.Module):
    def __init__(self, cross_attn, latent_transformer, depth, x_seqlen, x_dim, z_seqlen, d_model, out_dim, device):
        super().__init__()
        self.cross_attn_unique = cross_attn # first cross attention block is unique (weights not shared with other cross-attention blocks)
        self.cross_attn_shared = deepcopy(cross_attn) # rest of the cross-attention blocks with shared weights 
        self.latent_transformer = latent_transformer # all latent transformer blocks share weights
        self.depth = depth 
        self.z = nn.Parameter(torch.rand(z_seqlen, d_model)) # randomly initialized latent code with z_dim = d_model 
        self.x_emb = nn.Linear(x_dim, d_model, bias=False)
        self.z_pos_emb = nn.Parameter(torch.rand(z_seqlen, d_model)) # learnt positional embeddings
        self.x_pos_emb = nn.Parameter(torch.rand(x_seqlen, d_model)) # learnt positional embeddings
        self.proj_head = nn.Linear(d_model, out_dim)
    def forward(self, x): # x.shape: [b, M, C]
        batch_size = x.shape[0]
        x = self.x_emb(x) + self.x_pos_emb # x.shape: [b, M, d_model]
        z = self.z + self.z_pos_emb # z.shape: [N, d_model]
        z = z.unsqueeze(0).expand(batch_size, -1, -1) # z.shape: [b, N, d_model]
        
        # forward prop through first cross-attention block 
        z = self.cross_attn_unique(x, z, mask_padding=None) # key = x, query = z, value = x
        # forward prop through first latent transformer block 
        z = self.latent_transformer(z, mask_padding=None, mask_causal=None)
        # forward prop through rest of the blocks
        for _ in range(self.depth-1):
            z = self.cross_attn_shared(x, z, mask_padding=None) # key = x, query = z, value = x
            z = self.latent_transformer(z, mask_padding=None, mask_causal=None)
        # average over latent dimension 
        z = z.mean(dim=1) # z.shape: [b, d_model]
        out = self.proj_head(z) # out.shape: [b, out_dim]
        return out 
    
# function to instantiate perceiver
def init_perceiver(depth, x_seqlen, x_dim, z_seqlen, z_dim, out_dim, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    attn_singleHead = MultiHeadAttention(1, d_model, d_model, d_model, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    latent_transformer = Encoder(encoder_layer, n_layers, d_model) # latent transformer is just the transformerr encoder = stacked encoder layers
    cross_attn = Perceiver_xattn(deepcopy(attn_singleHead), deepcopy(ff), d_model, dropout) # perceiver cross_attn block
    perceiver = Perceiver(cross_attn, latent_transformer, depth, x_seqlen, x_dim, z_seqlen, d_model, out_dim, device)
    # initialize params - Xavier initialization
    for p in perceiver.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return perceiver

# function to get test loss and accuracy
def get_test_loss_accuracy(perceiver, testloader, criterion):
    print('Testing...')
    pbar = tqdm(total=len(testloader))
    correct = 0
    total = 0
    with torch.no_grad():
        test_loss = 0
        for i, data in enumerate(testloader):
            imgs, labels = data[0].to(device),data[1].to(device)
            # flatten imgs to a sequence of pixels 
            imgs = imgs.flatten(start_dim=2, end_dim=3) # imgs.shape: [b, c, h*w]
            imgs = imgs.permute(0, 2, 1) # imgs.shape: [b, h*w, c]
            scores = perceiver(imgs)
            loss = criterion(scores, labels)
            _, predicted = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss
            pbar.update(1)
        test_accuracy = correct/float(total)
        test_loss = test_loss / len(testloader)
    pbar.close()
    return test_loss.item(), test_accuracy

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
    img_channels = 1 # M in paper fig.1
    z_dim = d_model # D in paper fig.1
    z_seqlen = 16 # N in paper fig.1 - so the compression is from [28*28, d_model] -> [z_seqlen, d_model]
    x_dim = img_channels # C in paper fig.1
    x_seqlen = img_size * img_size # M in paper fig.1
    out_dim = 10 # num_classes for MNIST
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
    num_epochs = 30
    num_evals_per_epoch = 1
    random_seed = 10

    checkpoint_path = 'ckpts/perceiver_mnist.pt' # path to a save and load checkpoint of the trained model
    resume_training_from_ckpt = True 

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init model
    perceiver = init_perceiver(depth, x_seqlen, x_dim, z_seqlen, z_dim, out_dim, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # init optimizer
    optimizer = torch.optim.AdamW(params=perceiver.parameters(), lr=lr)

    if resume_training_from_ckpt:
        perceiver, optimizer = load_ckpt(checkpoint_path, perceiver, optimizer=optimizer, device=device, mode='train')

    # loss function
    loss_fn = nn.CrossEntropyLoss()

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
    results_test_accuracy = []
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
            scores = perceiver(imgs)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i+1) % (len(trainloader) // num_evals_per_epoch) == 0:
                test_loss, test_accuracy = get_test_loss_accuracy(perceiver, testloader, loss_fn)
                # print('epoch:{} \t i:{} \t train_loss:{} \t test_loss:{} \t test_accuracy:{}'.format(epoch, i, running_loss/2000, test_loss/2000, test_accuracy))
                results_train_loss.append(running_loss / (len(trainloader) // num_evals_per_epoch))
                results_test_loss.append(test_loss)
                results_test_accuracy.append(test_accuracy)
                running_loss = 0

            pbar.update(1)

        print('epoch_loss: ', epoch_loss / len(trainloader))
        print('test_accuracy: ', results_test_accuracy[-1])

        # save model checkpoint
        save_ckpt(device, checkpoint_path, perceiver, optimizer)

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
    plt.plot(x, results_test_accuracy, label='test_accuracy')
    plt.legend()
    plt.title('final_test_accuracy: ' + str(results_test_accuracy[-1]))
    plt.savefig('plots/perceiver_mnist_DataAug' + hyperparam_str + '.png')