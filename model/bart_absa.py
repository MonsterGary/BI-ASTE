import numpy as np
import torch
# from .modeling_bart import BartEncoder, BartDecoder, BartModel
# from transformers import BartTokenizer
# from fastNLP import seq_len_to_mask
# from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
# import torch.nn.functional as F
# # from fastNLP.models import Seq2SeqModel
# from .seq2seq_model import Seq2SeqModel
# from torch import nn
from .bilstm_biaffine_gcn_temp import *
import math


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True, use_syn_embed_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        # label_ids = sorted(label_ids, reverse=False)
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        # 这里在pipe中设置了第0个位置是sos, 第一个位置是eos, 所以做一下映射。还需要把task_id给映射一下
        mapping = torch.LongTensor([0, 2]+label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)   #768
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))
            if use_syn_embed_mlp:
                self.syn_embed_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                 nn.Dropout(0.3),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=tokens.size(1)>2)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores

        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[0][:, -1]


def get_prompt_mask(tokens,bsz,maxlen,tgt_seq_len):
    masks=[]
    #1--aspect: 2--opinion: 3--sentiment:   4--SSEP 5--category
    # mask.extend([0])
    if tokens[0][1]==5:
        for i in range(bsz):
            mask = []
            for j in range(tgt_seq_len[i] // 11):
            #for j in range(tgt_seq_len[i] //9):
                mask.extend([1, 0, 0, 2, 0, 0, 5,0,3, 0, 4])
                #mask.extend([1, 0, 0, 2, 0, 0, 3, 0, 4])
            if len(mask)!=maxlen-1:
                mask=mask+[0]*((maxlen-1)-len(mask))
            masks.append(mask)
    else:
        for i in range(bsz):
            mask = []
            for j in range(tgt_seq_len[i] // 11):
            #for j in range(tgt_seq_len[i] // 9):
                mask.extend([2, 0, 0, 1, 0, 0, 5,0,3, 0, 4])
                #mask.extend([2, 0, 0, 1, 0, 0, 3, 0, 4])
                # mask.extend([1, 0, 0, 2, 0, 0, 3, 0, 4])
            if len(mask) != maxlen - 1:
                mask = mask + [0] * ((maxlen - 1) - len(mask))
            masks.append(mask)
    masks=torch.tensor(masks).to('cuda')
    return masks

class CaGFBartDecoder(FBartDecoder):
    # Copy and generate, 计算score的时候，同时加上直接生成的分数
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=None, use_encoder_mlp=False,
                 use_syn_embed_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp,
                         use_syn_embed_mlp=use_syn_embed_mlp)
        self.use_syn_embed_mlp = use_syn_embed_mlp
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        # 否则是计算完分数，分数平均
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, tokens, state):
        # target tokens , state
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)#以固定模板mask prompt
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # prompt_mask = get_prompt_mask(tokens,bsz,max_len)

        # 把输入做一下映射
        # prompt_mapped_tokens= torch.where(prompt_mask==1,tokens,0)
        # prompt_mapped_tokens = torch.where(prompt_mapped_tokens == 1, 2, prompt_mapped_tokens)  #映射eos
        # tokens=torch.where(prompt_mask==1,0,tokens)
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)#这里要把prompt mask掉
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        # tokens = torch.where(prompt_mask==1,prompt_mapped_tokens,tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数

        if self.training:
            '''
            删除最后一个标记，该标记通常是序列结束（EOS）标记。
            这样做是因为在训练期间，通常训练模型以预测给定先前令牌的序列中的下一个令牌。在这种情况下，最后一个令牌没有要预测的后续令牌，因此它被从输入序列中删除。
            这在许多序列到序列模型中是一种常见的做法，其中目标序列向右移动一个位置，并且从输入序列中移除最后一个令牌，
            以确保模型不被训练来预测序列结束令牌作为序列中的下一个令牌。


            '''
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=tokens.size(1) > 3)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size  8*18*768
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)  # 填充-1e24 ，logits[:, :, 0]全是-1e24

        # 首先计算的是
        eos_scores = F.linear(hidden_state,
                              self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        # tags_embedding
        tag_scores = F.linear(hidden_state, self.dropout_layer(
            self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来

        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        syn_embed_out = state.syn_embed_out
        # biaffine_edge_relu = state.biaffine_edge_relu
        b, l, h = hidden_state.shape

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)
            if hasattr(self, 'syn_embed_mlp'):
                syn_embed_out = self.syn_embed_mlp(syn_embed_out)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size
        input_embed = self.dropout_layer(input_embed)
        if self.avg_feature:  # 先把feature合并一下
            # src_outputs = (0.2*src_outputs + 0.8*input_embed)
            src_outputs = (0.5 * src_outputs + 0.5 * input_embed)
            # src_outputs = (src_outputs + input_embed + syn_embed_out) / 3
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            syn_embed_scores = torch.einsum('blh,bnh->bln', hidden_state, syn_embed_out)  # bsz x max_len x max_word_len

            # biaffine_edge_scores = biaffine_feature[:,:l, :maxlen ]
            if self.use_syn_embed_mlp:
                word_scores = (gen_scores + word_scores + syn_embed_scores) / 3
            else:
                word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        # 填充-1e24 ，logits[:, :, 0]全是-1e24
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        # logits[:, :, 2:4].fill_(-1e32)  # 直接吧task的位置设置为不需要吧
        logits[:, :, self.src_start_index:] = word_scores

        return logits,hidden_state

class BartmarkSeq2SeqModel(Seq2SeqModel):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__(encoder, decoder)
        self.d_model=768
        self.aspect_fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.opinion_fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.proj = nn.Linear(2 * self.d_model, self.d_model, bias=True)
        self.classifier = nn.Linear(self.d_model, 3, bias=True)
        self.senti_classifier=nn.Linear(self.d_model, 3, bias=True)
        self.tripsenti_classifier = nn.Linear(self.d_model*3, 3, bias=True)
        self.epoch = 0
        self.k = 1
        # self.k_schedule = "linear"
        # self.k_schedule = "step"
        self.k_schedule = "constant_k"
        self.n_epochs = 0
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False, use_dual_encoder=False, use_syn_embed_mlp=False):
        # model = BartModel.from_pretrained('huggingface/bart-large',cache_dir='./huggingface/bart-large',force_download=True,resume_download=True)
        model = BartModel.from_pretrained('huggingface/model',cache_dir='./huggingface/model',force_download=True,resume_download=True)
        # model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens+17)
        encoder = model.encoder
        decoder = model.decoder
        encoder.use_dual_encoder = use_dual_encoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed
        pos_tokens = ['<<ADJ>>',
                      '<<ADP>>',
                      '<<ADV>>',
                      '<<AUX>>',
                      '<<CCONJ>>',
                      '<<DET>>',
                      '<<INTJ>>',
                      '<<NOUN>>',
                      '<<NUM>>',
                      '<<PART>>',
                      '<<PRON>>',
                      '<<PROPN>>',
                      '<<PUNCT>>',
                      '<<SCONJ>>',
                      '<<SYM>>',
                      '<<VERB>>',
                      '<<X>>'
                      ]
        for i in pos_tokens:
            indexes = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i))
            # embed = model.encoder.embed_tokens.weight.data[indexes[0]]
            # model.decoder.embed_tokens.weight.data[indexes[0]] = torch.zeros(768)

        if use_dual_encoder:
            encoder = DualBartEncoder(encoder, tokenizer)
        else:
            encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type =='avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                          avg_feature=True, use_encoder_mlp=use_encoder_mlp, use_syn_embed_mlp=use_syn_embed_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)
    # 通过CallBack设置epoch
    def set_epoch(self, epoch,n_epoch):
        self.epoch = epoch
        self.n_epochs = n_epoch
        threshold = int(self.n_epochs * 0.5)
        if self.k_schedule == "constant_k":
            self.k = 1
        elif self.epoch < threshold:
            self.k = 0
        else:
            if self.k_schedule == "linear":
                self.k = (self.epoch - threshold) / (self.n_epochs - threshold)
            elif self.k_schedule == "step":
                self.k = 1
            else:
                self.k = 0

    def prepare_state(self, head,pos,senti_value, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None,
        word_index=None,head_len=None,word_pair_deprel=None, matrix_mask=None, deprel_ids=None,tree_position=None,pair_position=None):
        if self.encoder.use_dual_encoder:
            # ===============================================================================here to modify
            # 添加bor，eor
            # 将src_tokens的eos和bos换成1和2，src_tokens是一个三维矩阵
            # bor_ids = torch.tensor([self.encoder.tokenizer.convert_tokens_to_ids('<unk>')]).to('cuda')
            # eor_ids = torch.tensor([self.encoder.tokenizer.convert_tokens_to_ids('<unk>')]).to('cuda')
            # syntactics = []
            # for i in range(src_tokens.shape[0]):
            #     # syntactic = src_tokens[i][1:src_seq_len[i] - 1].to('cuda')
            #     syntactic = pos[i][:src_seq_len[i]-2-3].to('cuda')
            #     syntactic = torch.cat((bor_ids, syntactic, eor_ids, src_tokens[i][src_seq_len[i]:].to('cuda')), dim=0)
            #     syntactics.append(syntactic)
            # syntactics = torch.stack(syntactics)
            syntactics = []
            for i in range(src_tokens.shape[0]):
                # syntactic = src_tokens[i][1:src_seq_len[i] - 1].to('cuda')
                syntactic = pos[i]
                # syntactic = torch.cat((bor_ids, syntactic, eor_ids, src_tokens[i][src_seq_len[i]:].to('cuda')), dim=0)
                syntactics.append(syntactic)
            syntactics = torch.stack(syntactics)

            # syntactics=src_tokens
            # attention_mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
            attention_mask_ori = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
            pos_attention_mask = seq_len_to_mask(head_len, max_len=max(head_len))
            # attention_mask_syntactic = attention_mask_ori
            # attention_mask = torch.cat((attention_mask_syntactic, attention_mask_ori), dim=1)
            encoder_mask = attention_mask_ori
            # print(attention_mask==attention_mask_ii)
            # 此处attention_mask应为src_tokens两倍
            # dict = self.encoder(src_tokens, syntactic,
            #                     attention_mask=encoder_mask,
            #                     output_hidden_states=True,
            #                     return_dict=True)
            #
            # encoder_outputs = dict.last_hidden_state
            # hidden_states = dict.hidden_states
            #
            # '''
            # 在BART模型所基于的Hugging Face Transformers库中，
            # hidden_states元组的长度始终为num_layers+1，其中num_layeres是模型中的层数。
            # 元组中的额外元素hidden_states[0]对应于模型的输入嵌入。
            # '''

            encoder_outputs, syn_embed_out,hidden_states, all_attentions, biaffine_edge_relu, nonlinear_outputs = self.encoder(
                src_tokens, syntactics,senti_value,
                src_seq_len,
                head,
                word_pair_deprel=word_pair_deprel,matrix_mask=matrix_mask,pos_attention_mask=pos_attention_mask,
                deprel_ids=deprel_ids,tree_position=tree_position,pair_position=pair_position,
                word_index=word_index, head_len=head_len,
                attention_mask=encoder_mask,
                output_hidden_states=True,
                return_dict=False)

            src_embed_outputs = hidden_states
            # state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
            # ================================================================================================这里有大问题===================================================================
            # syn_embed_out = src_embed_outputs[:, :len(src_tokens[0])-3, :]
            blank = torch.zeros(src_tokens.shape[0],1,768).to('cuda')
            # blank47 = torch.zeros(src_tokens.shape[0], 47, 768).to('cuda')
            syn_embed_out=torch.cat((blank,blank,blank,blank,syn_embed_out,blank),dim=1)
            blank_commons=torch.zeros(src_tokens.shape[0], (hidden_states.shape[1]-syn_embed_out.shape[1]), 768).to('cuda')
            syn_embed_out = torch.cat((syn_embed_out, blank_commons), dim=1)
            modified_encoder_out = encoder_outputs + nonlinear_outputs *syn_embed_out*self.k
            # modified_encoder_out = encoder_outputs + 0.4* syn_embed_out
            # state = BartState(encoder_outputs[:,-len(src_tokens[0]):,:], encoder_mask[:,-len(src_tokens[0]):], src_tokens, first, src_embed_outputs[:,-len(src_tokens[0]):,:], syn_embed_out, biaffine_edge_relu)
            state = BartState(modified_encoder_out, encoder_mask, src_tokens, first,
                              src_embed_outputs, syn_embed_out, biaffine_edge_relu)
            # setattr(state, 'tgt_seq_len', tgt_seq_len)

            return state, biaffine_edge_relu
        else:
            encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
            '''
            在BART模型所基于的Hugging Face Transformers库中，
            hidden_states元组的长度始终为num_layers+1，其中num_layeres是模型中的层数。
            元组中的额外元素hidden_states[0]对应于模型的输入嵌入。
            '''
            src_embed_outputs = hidden_states[0]
            state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
            return state

    def forward(self, head,pos_tag,senti_value,src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first,word_index,aspect_label,opinion_label,
                head_len=None,word_pair_deprel=None, matrix_mask=None, deprel_ids=None,tree_position=None,pair_position=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state, biaffine_edge_relu = self.prepare_state(head,pos_tag,senti_value, src_tokens, src_seq_len, first, tgt_seq_len,
                                                       word_index=word_index, head_len=head_len,
                                                       word_pair_deprel=word_pair_deprel,
                                                       matrix_mask=matrix_mask, deprel_ids=deprel_ids,
                                                       tree_position=tree_position, pair_position=pair_position)
        decoder_output,decoder_hidden = self.decoder(tgt_tokens, state)


        encoder_hidden=state.encoder_output[:,4:4+senti_value.shape[1]-2,:]     #只保留src_word bos eos已去掉
        res=[]
        word_mask=[]
        newwords=[]
        max_word_len=0
        for i in range(len(src_seq_len)):
            max_word_len = torch.max(word_index)
            temp_pos_mask = word_index[
                                i] - 1  # pos_mask从1开始数，第一个位置是第一个单词，所以统一减1，正好变成[0,1,2..., -2,-2]，其中-2是-1padding减1的结果
            trans_mask = torch.zeros((torch.max(word_index), word_index[i].size()[0])).to('cuda')
            # 利用矩阵乘法拷贝被分为多个token的word_pos
            for j in range(len(temp_pos_mask)):
                if temp_pos_mask[j] >= 0:
                    trans_mask[temp_pos_mask[j], j] = 1.0
            # pad到同样的维度，以方便后续stack
            newword=torch.matmul(trans_mask,encoder_hidden[i])
            res.append(trans_mask)
            newwords.append(newword)
        for ii in range(len(src_seq_len)):
            temp_pos_mask = word_index[
                                ii] - 1  # pos_mask从1开始数，第一个位置是第一个单词，所以统一减1，正好变成[0,1,2..., -2,-2]，其中-2是-1padding减1的结果
            word_maski=torch.arange(0,max_word_len).to('cuda')
            word_maski = torch.where(word_maski<=max(temp_pos_mask),1,0)
            word_mask.append(word_maski)

        word_index = torch.stack(res, dim=0)
        word_masks = torch.stack(word_mask,dim=0)
        # encoder_hidden = get_word_representation(word_index, encoder_hidden)     #只保留src_word bos eos已去掉
        encoder_hidden=torch.stack(newwords,dim=0)

        #token_null = [[50285]] * encoder_hidden.shape[0]
        #token_null_mask = [[1]] * encoder_hidden.shape[0]
        #token_null_mask=torch.tensor(token_null_mask).to('cuda')

        #token_null=self.encoder.embed_tokens(torch.tensor(token_null).to('cuda')) #<<NULL>>--50285
        #encoder_hidden=torch.cat((token_null,encoder_hidden),dim=1)
        #word_masks=torch.cat((token_null_mask,word_masks),dim=1)



        aspect_hidden_states = self.aspect_fc(encoder_hidden)
        opinion_hidden_states = self.opinion_fc(encoder_hidden)
        marker_position=get_prompt_mask(tgt_tokens,len(tgt_seq_len),max(tgt_seq_len),tgt_seq_len)
        decoder_marker_loss = self.marker_decoder_similarity(
           aspect_hidden_states, opinion_hidden_states,
           decoder_hidden,aspect_label,opinion_label,word_masks,marker_position,tgt_tokens
        )

        #decoder_marker_loss=0
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}, decoder_marker_loss
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def marker_decoder_similarity(self, aspect_hidden_state, opinion_hidden_state,
                                  decoder_hidden_state,aspect_label,opinion_label,
                                  word_masks,marker_position,target_seq):
        loss = 0
        batch = decoder_hidden_state.shape[0]
        # aspect
        if "A" in 'AO':
            aspect_marker, aspect_len = get_marker_state(1, marker_position, decoder_hidden_state)
            max_marker_per_sample = torch.max(aspect_len)
            encoder_index = torch.lt(
                torch.arange(max_marker_per_sample.int()).repeat(batch, 1).to(aspect_marker),
                aspect_len.repeat(1, max_marker_per_sample)
            )
            aspect_hidden_state = aspect_hidden_state.unsqueeze(1).repeat(1, max_marker_per_sample, 1, 1)
            aspect_hidden_state = aspect_hidden_state[encoder_index]
            aspect_label = aspect_label[encoder_index]
            aspect_mask = word_masks.unsqueeze(1).repeat(1, max_marker_per_sample, 1)
            aspect_mask = aspect_mask[encoder_index]
            aux_aspect_loss, aspect_probs = self.calculate_sim_loss(
                aspect_hidden_state, aspect_marker, aspect_label, aspect_mask
            )
            loss += aux_aspect_loss
        # opinion
        if "O" in 'AO':
            opinion_marker, opinion_len = get_marker_state(2, marker_position, decoder_hidden_state)
            max_marker_per_sample = torch.max(opinion_len)
            encoder_index = torch.lt(
                torch.arange(max_marker_per_sample.int()).repeat(batch, 1).to(opinion_marker),
                opinion_len.repeat(1, max_marker_per_sample)
            )
            opinion_hidden_state = opinion_hidden_state.unsqueeze(1).repeat(1, max_marker_per_sample, 1, 1)
            opinion_hidden_state = opinion_hidden_state[encoder_index]
            opinion_label = opinion_label[encoder_index]
            opinion_mask = word_masks.unsqueeze(1).repeat(1, max_marker_per_sample, 1)
            opinion_mask = opinion_mask[encoder_index]
            aux_opinion_loss, opinion_probs = self.calculate_sim_loss(
                opinion_hidden_state, opinion_marker, opinion_label, opinion_mask
            )
            loss += aux_opinion_loss

        # sentiment_marker, sentiment_len = get_marker_state(3, marker_position, decoder_hidden_state)
        # assert max(sentiment_len) == max(opinion_len) and max(sentiment_len) == max(aspect_len)
        # trip_marker = torch.cat([aspect_marker,opinion_marker,sentiment_marker],dim=-1)
        # # senti = self.tokenizer("positive negative neutral", return_tensors='pt')
        # # sentiment_hidden_state=self.embed_tokens(senti)
        # gold_senti = torch.tensor(self.map_emotion_labels(target_seq.view(-1))).to('cuda')
        # pred_senti = self.senti_classifier(sentiment_marker).view(-1, 3)
        # #pred_senti=self.tripsenti_classifier(trip_marker).view(-1, 3)
        # aux_senti_loss = F.cross_entropy(pred_senti, gold_senti.view(-1), reduction='none')
        # aux_senti_loss = torch.sum(aux_senti_loss) / len(gold_senti)
        # loss += aux_senti_loss
        return loss
    def calculate_sim_loss(self, marker_encoder_state, marker, label, mask):
        marker = marker.unsqueeze(-2).repeat(1, marker_encoder_state.shape[1], 1)
        state = torch.selu(self.proj(torch.cat([marker, marker_encoder_state], dim=-1)))
        state = self.classifier(state).view(-1, 3)
        aux_marker_loss = F.cross_entropy(state, label.view(-1).long(), reduction='none')
        aux_marker_loss = torch.sum(aux_marker_loss * mask.reshape(-1)) / torch.sum(mask)
        return aux_marker_loss, state
    def map_emotion_labels(self, target_seq):
        out=[]
        for seq in target_seq:
            if seq in range(2,5):
                out.append(seq-2)#隐射0-2
        return out



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs, syn_embed_out, biaffine_edge_relu):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.syn_embed_out = syn_embed_out
        self.biaffine_edge_relu = biaffine_edge_relu

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new

        self.syn_embed_out = self._reorder_state(self.syn_embed_out, indices)
        self.biaffine_edge_relu = self._reorder_state(self.biaffine_edge_relu, indices)

def get_word_representation(matrix, states):
    length = matrix.size(1)
    min_value = torch.min(states).item()
    states = states.unsqueeze(1).expand(-1, length, -1, -1)
    states = torch.masked_fill(states, matrix.eq(0).unsqueeze(-1), min_value)
    word_reps, _ = torch.max(states, dim=2)
    # word_reps = torch.relu(F.dropout(word_reps, p=0.1, training=self.training))
    return word_reps

def get_marker_state(index, marker_position, decoder_hidden_state):
    assert index in [1, 2, 3]  # 1-aspect, 2-opinion, 3-sentiment
    aos_marker_position = marker_position.eq(index)
    aos_marker_len = torch.sum(aos_marker_position, dim=-1).unsqueeze(-1)
    aos_marker = decoder_hidden_state[aos_marker_position.bool()]
    return aos_marker, aos_marker_len





