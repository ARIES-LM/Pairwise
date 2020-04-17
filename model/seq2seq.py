import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from scipy.stats import stats

from model.encoder import Encoder, EncoderLayer
from model.generator import Beam
from data.data import DocField, DocDataset, DocIter

import itertools


class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()

        self.emb_dp = args.input_drop_ratio
        self.model_dp = args.drop_ratio

        self.d_emb = args.d_emb

        self.sen_enc_type = args.senenc
        self.src_embed = nn.Embedding(args.doc_vocab, self.d_emb)

        h_dim = args.d_rnn
        d_mlp = args.d_mlp

        # sentence encoder
        self.sen_enc = nn.LSTM(self.d_emb, args.d_rnn // 2, bidirectional=True, batch_first=True)
        selfatt_layer = EncoderLayer(h_dim, 4, 512, args.attdp)
        self.encoder = Encoder(selfatt_layer, args.gnnl)

        self.decoder = nn.LSTM(h_dim, h_dim, batch_first=True)

        # pointer net
        self.linears = nn.ModuleList([nn.Linear(h_dim, d_mlp, False),
                                      nn.Linear(h_dim, d_mlp, False), nn.Linear(d_mlp, 1, False)])

        self.critic = None

        labelemb_dim = args.d_label
        d_pair = args.d_pair
        self.lamb = args.lamb_rela

        # future ffn
        self.future = nn.Sequential(nn.Linear(h_dim * 2, h_dim*2, False), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(h_dim*2, d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        self.w3 = nn.Linear(d_pair, 2, False)
        self.hist_left1 = nn.Sequential(nn.Linear(h_dim * 2, h_dim*2, False), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(h_dim*2, d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        # for sind, l2 half dim
        self.hist_left2 = nn.Sequential(nn.Linear(h_dim * 2, h_dim*2, False), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(h_dim*2, d_pair, False), nn.ReLU(), nn.Dropout(0.1))
        self.wleft1 = nn.Linear(d_pair, 2, False)
        self.wleft2 = nn.Linear(d_pair, 2, False)

        # new key
        d_pair_posi = d_pair + labelemb_dim
        self.pw_k = nn.Linear(d_pair_posi * 4, h_dim, False)

        self.pw_e = nn.Linear(h_dim, 1, False)

    def equip(self, critic):
        self.critic = critic

    def encode_history(self, paragraph, g):
        batch, num, hdim = paragraph.size()

        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)

        input = torch.cat((para_unq2, para_unq1), -1)
        rela_left1 = self.hist_left1(input)
        rela_left2 = self.hist_left2(input)
        return rela_left1, rela_left2

    def rela_encode(self, paragraph, g):
        batch, num, hdim = paragraph.size()
        # B N 1 H
        para_unq2 = paragraph.unsqueeze(2).expand(batch, num, num, hdim)
        # B 1 N H
        para_unq1 = paragraph.unsqueeze(1).expand(batch, num, num, hdim)
        # B N N H
        input = torch.cat((para_unq2, para_unq1), -1)
        return self.future(input)

    def rela_pred(self, paragraph, g):
        rela_vec = self.rela_encode(paragraph, g)
        rela_p = F.softmax(self.w3(rela_vec), -1)
        rela_vec_diret = torch.cat((rela_vec, rela_p), -1)

        hist_left1, hist_left2 = self.encode_history(paragraph, g)
        left1_p = F.softmax(self.wleft1(hist_left1), -1)
        left2_p = F.softmax(self.wleft2(hist_left2), -1)

        hist_vec_left1 = torch.cat((hist_left1, left1_p), -1)
        hist_vec_left2 = torch.cat((hist_left2, left2_p), -1)

        # prob, label = torch.topk(rela_p, 1)
        return (left1_p, left2_p, rela_p), rela_vec_diret, hist_vec_left1, hist_vec_left2

    def key(self, paragraph, rela_vec):
        rela_mask = rela_vec.new_ones(rela_vec.size(0), rela_vec.size(1), rela_vec.size(2)) \
                    - torch.eye(rela_vec.size(1)).cuda().unsqueeze(0)

        rela_vec_mean = torch.sum(rela_vec * rela_mask.unsqueeze(3), 2) / rela_mask.sum(2, True)
        pre_key = torch.cat((paragraph, rela_vec_mean), -1)
        key = self.linears[1](pre_key)
        return key

    def forward(self, src_and_len, tgt_and_len, doc_num):
        document_matrix, _, hcn = self.encode(src_and_len, doc_num)
        target, tgt_len = tgt_and_len
        batch, num = target.size()

        tgt_len_less = tgt_len
        target_less = target

        target_mask = torch.zeros_like(target_less).byte()
        pointed_mask_by_target = torch.zeros_like(target).byte()

        # *************
        # relative order loss
        rela_vec = self.rela_encode(document_matrix, hcn[0])
        score = self.w3(rela_vec)

        # B N N 2
        logp_rela = F.log_softmax(score, -1)

        truth = torch.tril(logp_rela.new_ones(num, num)).long().unsqueeze(0).expand(batch, num, num)

        logp_rela = logp_rela[torch.arange(batch).unsqueeze(1), target]
        logp_rela = logp_rela[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                              torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        loss_rela = self.critic(logp_rela.view(-1, 2), truth.contiguous().view(-1))

        # history loss
        rela_hist_left1, rela_hist_left2 = self.encode_history(document_matrix, hcn[0])
        score_left1 = self.wleft1(rela_hist_left1)
        score_left2 = self.wleft2(rela_hist_left2)

        logp_left1 = F.log_softmax(score_left1, -1)
        logp_left2 = F.log_softmax(score_left2, -1)

        logp_left1 = logp_left1[torch.arange(batch).unsqueeze(1), target]
        logp_left1 = logp_left1[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                                torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        logp_left2 = logp_left2[torch.arange(batch).unsqueeze(1), target]
        logp_left2 = logp_left2[torch.arange(batch).unsqueeze(1).unsqueeze(2),
                                torch.arange(num).unsqueeze(0).unsqueeze(2), target.unsqueeze(1)]

        loss_left1_mask = torch.tril(target.new_ones(num, num), -1).unsqueeze(0).expand(batch, num, num)
        truth_left1 = loss_left1_mask - torch.tril(target.new_ones(num, num), -2).unsqueeze(0)

        loss_left2_mask = torch.tril(target.new_ones(num, num), -2).unsqueeze(0).expand(batch, num, num)
        truth_left2 = loss_left2_mask - torch.tril(target.new_ones(num, num), -3).unsqueeze(0)

        loss_left1 = self.critic(logp_left1.view(-1, 2), truth_left1.contiguous().view(-1))
        loss_left2 = self.critic(logp_left2.view(-1, 2), truth_left2.contiguous().view(-1))

        eye_mask = torch.eye(num).byte().cuda().unsqueeze(0)
        rela_mask = torch.ones_like(truth_left1).byte() - eye_mask

        left1_mask = loss_left1_mask.clone()
        left2_mask = loss_left1_mask.clone()

        for b in range(batch):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len_less[b]] = 1

            rela_mask[b, tgt_len[b]:] = 0
            rela_mask[b, :, tgt_len[b]:] = 0

            left1_mask[b, tgt_len[b]:] = 0
            left2_mask[b, tgt_len[b]:] = 0

            if tgt_len[b] >= 4:
                for ix in range(3, tgt_len[b]):
                    weight = document_matrix.new_ones(ix)
                    weight[-1] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left1_mask[b, ix, negix] = 0

                    weight[-1] = 1
                    weight[-2] = 0
                    negix = torch.multinomial(weight, ix - 1 - 1)
                    left2_mask[b, ix, negix] = 0

        loss_rela.masked_fill_(rela_mask.view(-1) == 0, 0)
        loss_rela = loss_rela.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_rela = loss_rela.sum() / batch


        loss_left1.masked_fill_(left1_mask.view(-1) == 0, 0)
        loss_left1 = loss_left1.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_left1 = loss_left1.sum() / batch

        loss_left2.masked_fill_(left2_mask.view(-1) == 0, 0)
        loss_left2 = loss_left2.view(batch, num, -1).sum(2) / target_mask.sum(1, True).float()
        loss_left2 = loss_left2.sum() / batch

        # *************

        # B N-2 H
        dec_inputs = document_matrix[torch.arange(document_matrix.size(0)).unsqueeze(1), target_less[:, :-1]]
        start = dec_inputs.new_zeros(batch, 1, dec_inputs.size(2))
        # B N-1 H
        dec_inputs = torch.cat((start, dec_inputs), 1)

        p_direc = F.softmax(score, -1)
        rela_vec_diret = torch.cat((rela_vec, p_direc), -1)
        p_left1 = F.softmax(score_left1, -1)
        p_left2 = F.softmax(score_left2, -1)

        hist_vec_left1 = torch.cat((rela_hist_left1, p_left1), -1)
        hist_vec_left2 = torch.cat((rela_hist_left2, p_left2), -1)

        dec_outputs = []
        pw_keys = []
        # mask already pointed nodes
        pointed_mask = [rela_mask.new_zeros(batch, 1, num)]

        eye_zeros = torch.ones_like(eye_mask) - eye_mask
        eye_zeros = eye_zeros.unsqueeze(-1)

        for t in range(num):
            if t == 0:
                rela_mask = rela_mask.unsqueeze(-1)
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)
            else:
                # B (left1)
                tar = target[:, t - 1]

                # future
                rela_mask[torch.arange(batch), tar] = 0
                rela_mask[torch.arange(batch), :, tar] = 0

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(batch), :, tar] = 1
                if t > 1:
                    l2_mask[torch.arange(batch), :, target[:, t - 2]] = 1

                pm = pointed_mask[-1].clone().detach()
                pm[torch.arange(batch), :, tar] = 1
                pointed_mask.append(pm)

            # history information
            cur_hist_l1 = hist_vec_left1.masked_fill(l1_mask == 0, 0).sum(2)
            cur_hist_l2 = hist_vec_left2.masked_fill(l2_mask == 0, 0).sum(2)

            # future information
            rela_vec_diret.masked_fill_(rela_mask == 0, 0)
            forw_pw = rela_vec_diret.mean(2)
            back_pw = rela_vec_diret.mean(1)

            pw_info = torch.cat((cur_hist_l1, cur_hist_l2, forw_pw, back_pw), -1)
            pw_key = self.pw_k(pw_info)
            pw_keys.append(pw_key.unsqueeze(1))

            dec_inp = dec_inputs[:, t:t + 1]

            # B 1 H
            output, hcn = self.decoder(dec_inp, hcn)
            dec_outputs.append(output)

        # B N-1 H
        dec_outputs = torch.cat(dec_outputs, 1)
        # B N-1 1 H
        query = self.linears[0](dec_outputs).unsqueeze(2)

        key = torch.cat(pw_keys, 1)
        # B N-1 N H
        e = torch.tanh(query + key)
        # B N-1 N
        e = self.linears[2](e).squeeze(-1)

        # B N-1 N
        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)

        logp = F.log_softmax(e, dim=-1)
        logp = logp.view(-1, logp.size(-1))
        loss = self.critic(logp, target_less.contiguous().view(-1))

        loss.masked_fill_(target_mask.view(-1) == 0, 0)
        loss = loss.sum() / batch

        total_loss = loss + (loss_rela + loss_left1 + loss_left2) * self.lamb
        if torch.isnan(total_loss):
            exit('nan')
        return total_loss, loss, loss_rela * self.lamb, loss_left1 * self.lamb, loss_left2 * self.lamb

    def bow(self, src_and_len, doc_num, position=False):
        src, length = src_and_len
        x = self.src_embed(src)

        if len(doc_num) > 1:
            mask = torch.zeros_like(src).float()
            for i in range(len(mask)):
                mask[i][:length[i]] = 1
            mask = mask.unsqueeze(2)

            x = x * mask

            x = x.view(len(doc_num), 5, -1, self.d_emb)
            mask = mask.view(len(doc_num), 5, -1, 1).sum(2)

            xmean = torch.sum(x, 2) / mask
            xmax, _ = torch.max(x, 2)

            x = torch.cat((xmean, xmax), 2)
        else:
            x = x.view(1, 5, -1, self.d_emb)
            xmean = torch.mean(x, 2)
            xmax, _ = torch.max(x, 2)
            x = torch.cat((xmean, xmax), 2)

        return x

    def rnn_enc(self, src_and_len, doc_num):
        '''
        :param src_and_len:
        :param doc_num: B, each doc has sentences number
        :return: document matirx
        '''
        src, length = src_and_len

        sorted_len, ix = torch.sort(length, descending=True)
        sorted_src = src[ix]

        # bi-rnn must uses pack, else needs mask
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_src, sorted_len, batch_first=True)
        x = packed_x.data

        x = self.src_embed(x)

        if self.emb_dp > 0:
            x = F.dropout(x, self.emb_dp, self.training)

        packed_x = nn.utils.rnn.PackedSequence(x, packed_x.batch_sizes)

        # 2 TN H
        states, (hn, _) = self.sen_enc(packed_x)
        # states = nn.utils.rnn.pad_packed_sequence(states, True)

        # TN 2H
        hn = hn.transpose(0, 1).contiguous().view(src.size(0), -1)

        # hn = hn.squeeze(0)

        _, recovered_ix = torch.sort(ix, descending=False)
        hn = hn[recovered_ix]
        # states = states[recovered_ix]

        # max-pooling
        # hn, _ = states.max(1)

        batch_size = len(doc_num)
        maxdoclen = max(doc_num)
        output = hn.view(batch_size, maxdoclen, -1)

        return output

    def encode(self, src_and_len, doc_num):
        if self.sen_enc_type == 'rnn':
            sentences = self.rnn_enc(src_and_len, doc_num)
        else:
            sentences = self.bow(src_and_len, doc_num)

        if self.model_dp > 0:
            sentences = F.dropout(sentences, self.model_dp, self.training)
        # self-att
        batch = sentences.size(0)
        sents_mask = sentences.new_zeros(batch, sentences.size(1)).byte()

        for i in range(batch):
            sents_mask[i, :doc_num[i]] = 1
        # B 1 N
        sents_mask = sents_mask.unsqueeze(1)

        para = self.encoder(sentences, sents_mask)

        # key = self.linears[1](para)

        # B N 1
        sents_mask = sents_mask.squeeze(1).unsqueeze(2).float()
        para = para * sents_mask

        hn = para.sum(1) / sents_mask.sum(1)

        hn = hn.unsqueeze(0)
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)

        # key = self.linears[1](sentences)
        return sentences, para, hcn

    def rela_att(self, prev_h, rela, rela_k, rela_mask):
        # B 1 H
        q = self.rela_q(prev_h).transpose(0, 1)
        e = self.rela_e(torch.tanh(q + rela_k))

        e.masked_fill_(rela_mask == 0, -1e9)
        alpha = F.softmax(e, 1)
        context = torch.sum(alpha * rela, 1, True)
        return context

    def stepv2(self, prev_y, prev_handc, keys, mask, rela_vec, hist_left1, hist_left2, rela_mask, l1_mask, l2_mask):
        '''
        :param prev_y: (seq_len=B, 1, H)
        :param prev_handc: (1, B, H)
        :return:
        '''

        _, (h, c) = self.decoder(prev_y, prev_handc)
        # 1 B H-> B H-> B 1 H
        query = h.squeeze(0).unsqueeze(1)
        query = self.linears[0](query)

        # history
        left1 = hist_left1.masked_fill(l1_mask.unsqueeze(-1) == 0, 0).sum(2)
        left2 = hist_left2.masked_fill(l2_mask.unsqueeze(-1) == 0, 0).sum(2)

        # future
        rela_vec.masked_fill_(rela_mask.unsqueeze(-1) == 0, 0)
        forw_futu = rela_vec.mean(2)
        back_futu = rela_vec.mean(1)

        pw = torch.cat((left1, left2, forw_futu, back_futu), -1)
        keys = self.pw_k(pw)

        # B N H
        e = torch.tanh(query + keys)
        # B N
        e = self.linears[2](e).squeeze(2)
        e.masked_fill_(mask, -1e9)

        logp = F.log_softmax(e, dim=-1)
        return h, c, logp

    def load_pretrained_emb(self, emb):
        self.src_embed = nn.Embedding.from_pretrained(emb, freeze=False).cuda()


def beam_search_pointer(args, model, src_and_len, doc_num):
    sentences, _, dec_init = model.encode(src_and_len, doc_num)
    document = sentences.squeeze(0)
    T, H = document.size()

    keys = model.linears[1](sentences)

    # future
    rela_out, rela_vec, hist_left1, hist_left2 = model.rela_pred(sentences, dec_init[0])

    eye_mask = torch.eye(T).cuda().byte()
    eye_zeros = torch.ones_like(eye_mask) - eye_mask

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()

            rela_mask = eye_zeros.unsqueeze(0)

            l1_mask = torch.zeros_like(rela_mask)
            l2_mask = torch.zeros_like(rela_mask)
        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()
            # beam 1 H
            dec_input = document[index].unsqueeze(1)

            temp_batch = index.size(0)

            pointed_mask[torch.arange(temp_batch), index] = 1

            rela_mask[torch.arange(temp_batch), :, index] = 0
            rela_mask[torch.arange(temp_batch), index] = 0

            l1_mask = torch.zeros_like(rela_mask)
            l2_mask = torch.zeros_like(rela_mask)

            l1_mask[torch.arange(temp_batch), :, index] = 1
            if t > 1:
                left2_index = index.new_tensor(list(map(lambda cand: cand[-2], candidates)))
                l2_mask[torch.arange(temp_batch), :, left2_index] = 1

        dec_h, dec_c, log_prob = model.stepv2(dec_input, dec_init, keys, pointed_mask,
                                              rela_vec, hist_left1, hist_left2, rela_mask, l1_mask, l2_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src_and_len[0].new_tensor(remain_list)

        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        rela_mask = rela_mask.index_select(0, beam_remain_ix)
        rela_vec = rela_vec.index_select(0, beam_remain_ix)

        hist_left1 = hist_left1.index_select(0, beam_remain_ix)
        hist_left2 = hist_left2.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output, rela_out


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))


def train(args, train_iter, dev, test_real, fields, checkpoint):
    model = PointerNet(args)
    model.cuda()

    # 2.
    DOC, ORDER = fields
    print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    wd = 1e-5

    if args.optimizer == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    elif args.optimizer == 'Adadelta':
        opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)
    elif args.optimizer == 'AdaGrad':
        opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=wd)
    else:
        raise NotImplementedError

    best_score = -np.inf
    best_iter = 0
    offset = 0

    train_loss = []
    dev_loss = []

    # lr_sche = torch.optim.lr_scheduler.ExponentialLR(opt, args.lrdecay)

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()
    patience = args.patience

    early_stop = args.early_stop

    for epc in range(args.maximum_steps):

        for iters, batch in enumerate(train_iter):
            model.train()

            model.zero_grad()

            t1 = time.time()
            loss, point_loss, rela_loss, left1_loss, left2_loss = model(batch.doc, batch.order, batch.doc_len)

            loss.backward()

            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            t2 = time.time()
            if iters % 500 == 0:
                print('epc:{} iter:{} point:{:.2f} futu:{:.2f} '
                      'left1:{:.2f}, left2:{:.2f} t:{:.2f}'.format(epc, iters + 1,
                                                                   point_loss, rela_loss, left1_loss, left2_loss,
                                                                   t2 - t1))

        with torch.no_grad():
            print('valid..............')
            score, pmr, ktau = valid_model(args, model, dev, DOC)
            print('epc:{}, acc:{:.2%}, best:{:.2%}'.format(epc, score, best_score))

            if score > best_score:
                best_score = score
                best_iter = epc

                print('save best model at epc={}'.format(epc))
                checkpoint = {'model': model.state_dict(),
                              'optim': opt.state_dict(),
                              'args': args,
                              'best_score': best_score}
                torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

            if early_stop and (epc - best_iter) >= early_stop:
                print('early stop at epc {}'.format(epc))
                break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2%}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2%}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        acc, pmr, ktau = valid_model(args, model, test_real, DOC, shuflle_times=1)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2%}'.format(acc, pmr, ktau))


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0
        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n
        return sum(total_score) / number
    else:
        # f = open(args.writetrans, 'w')
        truth = []
        predicted = []

        for j, dev_batch in enumerate(dev):
            tru = dev_batch.order[0].view(-1)
            truth.append(tru.tolist())

            pred, out = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len)
            predicted.append(pred)
            # print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
            #   file=f)

        right, total = 0, 0
        pmr_right = 0
        taus = []
        import itertools
        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
            eq = np.equal(t, p)
            right += eq.sum()
            total += len(t)

            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])

            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            taus.append(tau)

        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                             list(itertools.chain.from_iterable(predicted)))
        pmr = pmr_right / len(truth)
        taus = np.mean(taus)
        # f.close()
        return acc, pmr, taus


def decode(args, test_real, fields, checkpoint):
    with torch.no_grad():
        model = PointerNet(args)
        model.cuda()
        DOC, ORDER = fields
        print('load parameters')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        acc, pmr, ktau = valid_model(args, model, test_real, DOC)
        print('test acc:{:.2%} pmr:{:.2%} ktau:{:.2f}'.format(acc, pmr, ktau))

