import torch
import pandas as pd
import numpy as np

ALPHABET = list('ARNDCQEGHILKMFPSTWYV')
ALPHABET = {ALPHABET[i]:i for i in range(len(ALPHABET))}

def embed_genes(gene_strs, gene_to_ind):
    indices = torch.LongTensor([gene_to_ind[entry] for entry in gene_strs])
    one_hot = torch.nn.functional.one_hot(indices, num_classes=len(gene_to_ind))
    return one_hot

def embed_seqs(seqs, pad_length=32, alph=ALPHABET):
    index_seqs = [torch.LongTensor([alph[letter] for letter in seq]) for seq in seqs]
    index_seqs.append(torch.zeros((pad_length)).long())
    padded = torch.nn.utils.rnn.pad_sequence(index_seqs,padding_value=len(alph))
    one_hot = torch.nn.functional.one_hot(padded, num_classes=len(alph)+1)[:, :, :len(alph)]
    one_hot = torch.permute(one_hot, (1,2,0))[:-1]
    return one_hot

def train_epoch_cell(model, device, train_dfs, test_tuple, optimizer, criterion, epoch, vj_genes, 
                        drop_heavy=False, batch_size=1000):
    model.train()
    (sl_test, dl_test_1, dl_test_2) = test_tuple

    train_scores = []

    for i in range(0, train_dfs[1].shape[0], batch_size):
        sl_subs = train_dfs[0].iloc[2*i:2*(i + batch_size)]
        dl_subs = train_dfs[1].iloc[i:i+batch_size]
        sl_batch = embed_sl(sl_subs, device, vj_genes, drop_heavy=drop_heavy)
        (dl_batch_1, dl_batch_2) = embed_dl(dl_subs, device, vj_genes, drop_heavy=drop_heavy)
        
        optimizer.zero_grad()
        
        output_sl = model(sl_batch[0], sl_batch[1], sl_batch[2])
        output_sl = (output_sl[:, 1] - output_sl[:, 0]).reshape(-1, 2).min(dim=1)[0]
        output_dl_1 = model(dl_batch_1[0], dl_batch_1[1], dl_batch_1[2])
        output_dl_1 = output_dl_1[:, 1] - output_dl_1[:, 0]
        output_dl_2 = model(dl_batch_2[0], dl_batch_2[1], dl_batch_2[2])
        output_dl_2 = output_dl_2[:, 1] - output_dl_2[:, 0]
        output_dl = torch.minimum(output_dl_1, output_dl_2)
        output = torch.cat((output_sl, output_dl))
        output = torch.stack((torch.zeros_like(output), output), dim=1)
        yi = torch.zeros((output.shape[0])).long().to(device)
        yi[:output_sl.shape[0]] = 1
        loss = criterion(output, yi)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_scores.append(loss.detach().cpu().numpy())
    train_loss = np.array(train_scores).mean()
    
    model.eval()
    output_sl = model(sl_test[0], sl_test[1], sl_test[2])
    output_sl = (output_sl[:, 1] - output_sl[:, 0]).reshape(-1, 2).min(dim=1)[0].detach().cpu().numpy()
    output_dl_1 = model(dl_test_1[0], dl_test_1[1], dl_test_1[2])
    output_dl_1 = output_dl_1[:, 1] - output_dl_1[:, 0]
    output_dl_2 = model(dl_test_2[0], dl_test_2[1], dl_test_2[2])
    output_dl_2 = output_dl_2[:, 1] - output_dl_2[:, 0]
    output_dl = torch.minimum(output_dl_1, output_dl_2).detach().cpu().numpy()

    test_recall = (output_dl < np.quantile(output_sl, 0.1)).mean()
    print('Epoch [{}], Train loss: {}, Test recall: {}'.format(epoch, train_loss, test_recall), end='')
    return test_recall

def embed_sl(sl_df, device, vj_genes, drop_heavy=False):
    (v_genes_h, j_genes_h, v_genes_l, j_genes_l) = vj_genes

    ## SL CDR3
    light = embed_seqs(sl_df.CDRL3.values, pad_length=32, alph=ALPHABET)
    heavy = embed_seqs(sl_df.CDRH3.values, pad_length=32, alph=ALPHABET)
    if drop_heavy:
        heavy = torch.zeros_like(light)
    sl_x = torch.stack([heavy, light], dim=2).float().to(device)

    ## SL VJ
    sl_vl = embed_genes(sl_df.v_gene_L, v_genes_l)
    sl_jl = embed_genes(sl_df.j_gene_L, j_genes_l)
    sl_vh = embed_genes(sl_df.v_gene_H, v_genes_h)
    sl_jh = embed_genes(sl_df.j_gene_H, j_genes_h)
    if drop_heavy:
        sl_vh = torch.zeros((sl_vl.shape[0], 0)).long()
        sl_jh = torch.zeros((sl_vl.shape[0], 0)).long()

    sl_v = torch.cat([sl_vh, sl_vl], dim=1).float().to(device)
    sl_j = torch.cat([sl_jh, sl_jl], dim=1).float().to(device)

    return(sl_x, sl_v, sl_j)

def embed_dl(dl_df, device, vj_genes, drop_heavy=False):
    (v_genes_h, j_genes_h, v_genes_l, j_genes_l) = vj_genes

    ## DL CDR3
    light_1 = embed_seqs(dl_df.CDRL3_1.values, pad_length=32, alph=ALPHABET)
    light_2 = embed_seqs(dl_df.CDRL3_2.values, pad_length=32, alph=ALPHABET)
    heavy = embed_seqs(dl_df.CDRH3.values, pad_length=32, alph=ALPHABET)
    if drop_heavy:
        heavy = torch.zeros_like(light_1)
    dl_x_1 = torch.stack([heavy, light_1], dim=2).float().to(device)
    dl_x_2 = torch.stack([heavy, light_2], dim=2).float().to(device)

    ## DL VJ
    dl_vl_1 = embed_genes(dl_df.v_gene_L_1, v_genes_l)
    dl_vl_2 = embed_genes(dl_df.v_gene_L_2, v_genes_l)
    dl_jl_1 = embed_genes(dl_df.j_gene_L_1, j_genes_l)
    dl_jl_2 = embed_genes(dl_df.j_gene_L_2, j_genes_l)
    dl_vh = embed_genes(dl_df.v_gene_H, v_genes_h)
    dl_jh = embed_genes(dl_df.j_gene_H, j_genes_h)
    if drop_heavy:
        dl_vh = torch.zeros((dl_vl_1.shape[0], 0)).long()
        dl_jh = torch.zeros((dl_vl_1.shape[0], 0)).long()

    dl_v_1 = torch.cat([dl_vh, dl_vl_1], dim=1).float().to(device)
    dl_v_2 = torch.cat([dl_vh, dl_vl_2], dim=1).float().to(device)
    dl_j_1 = torch.cat([dl_jh, dl_jl_1], dim=1).float().to(device)
    dl_j_2 = torch.cat([dl_jh, dl_jl_2], dim=1).float().to(device)

    return((dl_x_1, dl_v_1, dl_j_1), (dl_x_2, dl_v_2, dl_j_2))
