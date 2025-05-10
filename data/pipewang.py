from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key

def cmp(v1, v2):
    if v1['from']==v2['from']:
        return v1['to'] - v2['to']
    return v1['from'] - v2['from']

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


def get_trip_tgt(ins,opinion_first,cum_lens,_word_bpes,tokenizer,target_shift,mapping2targetid):
    target = [0]  # 特殊的开始
    target_spans = []
    generate_target_spans = []
    aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
    if opinion_first == 'O':
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
    elif opinion_first == 'A':
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))

    aspect_label=[]
    opinion_label=[]
    for aspects, opinions in aspects_opinions:  # 预测bpe的start
        aspect_ = [0] * (len(cum_lens)-2)  ##去掉bos eos
        opinion_ = [0] * (len(cum_lens)-2) # st==2   ed==1
        assert aspects['index'] == opinions['index']
        a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
        a_end_bpe = cum_lens[aspects['to'] - 1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
        o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
        o_end_bpe = cum_lens[opinions['to'] - 1]  # 因为有一个sos shift
        aspect_[aspects['from']]=2
        if aspects['from']!=aspects['to']-1:
            aspect_[aspects['to']-1]=1
        opinion_[opinions['from']] = 2
        if opinions['from']!=opinions['to'] - 1:
            opinion_[opinions['to']-1]=1
        aspect_label.append(aspect_)
        opinion_label.append(opinion_)
        # 这里需要evaluate是否是对齐的
        for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                             (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):
            assert _word_bpes[idx] == \
                   tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                   _word_bpes[idx] == \
                   tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

        if opinion_first=='O':
            generate_target_spans.append(
                [mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift,
                 mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift])
            target_spans.append(
                [o_start_bpe + target_shift, o_end_bpe + target_shift,
                 a_start_bpe + target_shift, a_end_bpe + target_shift])
        else:
            generate_target_spans.append(
                [mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift,
                 mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift, ])
            target_spans.append(
                [a_start_bpe + target_shift, a_end_bpe + target_shift,
                 o_start_bpe + target_shift, o_end_bpe + target_shift, ])
        generate_target_spans[-1].extend(
            [mapping2targetid['S'] + 2, mapping2targetid[aspects['polarity']] + 2])  # 前面有sos和eos
        generate_target_spans[-1].append(mapping2targetid['SEP'] + 2)
        generate_target_spans[-1] = tuple(generate_target_spans[-1])
        target_spans[-1].append(mapping2targetid[aspects['polarity']] + 2)  # 前面有sos和eos
        target_spans[-1] = tuple(target_spans[-1])
    target.extend(list(chain(*generate_target_spans)))
    target = target[:-1]  # 去掉最后一个ssep
    target.append(1)  # append 1是由于特殊的eos
    return target, target_spans,aspect_label,opinion_label
'''
def get_tgt(ins,opinion_first,cum_lens,_word_bpes,tokenizer,target_shift,mapping2targetid):
    target = [0]  # 特殊的开始
    a_target = [0]  # 特殊的开始
    o_target = [0]  # 特殊的开始
    target_spans = []
    aspect_label=[]
    opinion_label=[]
    aesc_target_spans = []
    ae_target_spans = []
    oe_target_spans = []
    aspects = sorted(ins['aspects'], key=cmp_to_key(cmp))
    opinions = sorted(ins['opinions'], key=cmp_to_key(cmp))
    # 以词的最后一个子词的token作为idx
    for aspect in aspects:
        s_bpe = cum_lens[aspect['from']] + target_shift
        e_bpe = cum_lens[aspect['to'] - 1] + target_shift
        polarity = mapping2targetid[aspect['polarity']]
        ae_target_spans.append((s_bpe, e_bpe))
        a_target.extend([mapping2targetid['A'] + 2,s_bpe, e_bpe, mapping2targetid['S'] + 2,polarity+2,mapping2targetid['SEP'] + 2])
        # sc_target_spans.append(polarity)
        aesc_target_spans.append((s_bpe, e_bpe, polarity+2))
    if a_target[-1] != 0:
        a_target = a_target[:-1]  # 去掉最后一个ssep
    a_target.append(1)  # append 1是由于特殊的eos
    for opinion in opinions:
        s_bpe = cum_lens[opinion['from']] + target_shift
        e_bpe = cum_lens[opinion['to'] - 1] + target_shift
        oe_target_spans.append((s_bpe, e_bpe))
        o_target.extend((mapping2targetid['O'] + 2,s_bpe, e_bpe,mapping2targetid['SEP'] + 2))
    if o_target[-1]!=0:
        o_target = o_target[:-1]  # 去掉最后一个ssep
    o_target.append(1)  # append 1是由于特殊的eos

    for aspect in aspects:
        aspect_ = [0] * (len(cum_lens)-2)  ##去掉bos eos
        a_start_bpe = cum_lens[aspect['from']]  # 因为有一个sos shift
        a_end_bpe = cum_lens[aspect['to'] - 1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
        aspect_[aspect['from']]=2
        if aspect['from']!=aspect['to']-1:
            aspect_[aspect['to']-1]=1
        aspect_label.append(aspect_)
    if len(aspects)==0:
        aspect_label.append([0] * (len(cum_lens)-2))
    for opinion in opinions:
        opinion_ = [0] * (len(cum_lens)-2) # st==2   ed==1
        o_start_bpe = cum_lens[opinion['from']]  # 因为有一个sos shift
        o_end_bpe = cum_lens[opinion['to'] - 1]  # 因为有一个sos shift
        opinion_[opinion['from']] = 2
        if opinion['from']!=opinion['to'] - 1:
            opinion_[opinion['to']-1]=1
        opinion_label.append(opinion_)
    if len(opinions)==0:
        opinion_label.append([0] * (len(cum_lens)-2))
    return a_target,o_target, aesc_target_spans,aspect_label,opinion_label
'''
def get_tgt(ins,opinion_first,cum_lens,_word_bpes,tokenizer,target_shift,mapping2targetid):
    target = [0]  # 特殊的开始
    target_spans = []
    generate_target_spans = []
    aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
    if opinion_first == 'O':
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
    elif opinion_first == 'A':
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))

    aspect_label=[]
    opinion_label=[]
    for aspects, opinions in aspects_opinions:  # 预测bpe的start
        aspect_ = [0] * (len(cum_lens)-2)  ##去掉bos eos
        opinion_ = [0] * (len(cum_lens)-2) # st==2   ed==1
        assert aspects['index'] == opinions['index']
        a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
        a_end_bpe = cum_lens[aspects['to'] - 1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
        o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
        o_end_bpe = cum_lens[opinions['to'] - 1]  # 因为有一个sos shift
        aspect_[aspects['from']]=2
        if aspects['from']!=aspects['to']-1:
            aspect_[aspects['to']-1]=1
        opinion_[opinions['from']] = 2
        if opinions['from']!=opinions['to'] - 1:
            opinion_[opinions['to']-1]=1
        aspect_label.append(aspect_)
        opinion_label.append(opinion_)
        # 这里需要evaluate是否是对齐的
        for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                             (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):
            assert _word_bpes[idx] == \
                   tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                   _word_bpes[idx] == \
                   tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

        if opinion_first=='O':
            generate_target_spans.append(
                [mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift,
                 mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift])
            target_spans.append(
                [o_start_bpe + target_shift, o_end_bpe + target_shift,
                 a_start_bpe + target_shift, a_end_bpe + target_shift])
        else:
            generate_target_spans.append(
                [mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift,
                 mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift, ])
            target_spans.append(
                [a_start_bpe + target_shift, a_end_bpe + target_shift,
                 o_start_bpe + target_shift, o_end_bpe + target_shift, ])
        # generate_target_spans[-1].extend(
        #     [mapping2targetid['S'] + 2, mapping2targetid[aspects['polarity']] + 2])  # 前面有sos和eos
        generate_target_spans[-1].append(mapping2targetid['SEP'] + 2)
        generate_target_spans[-1] = tuple(generate_target_spans[-1])
        # target_spans[-1].append(mapping2targetid[aspects['polarity']] + 2)  # 前面有sos和eos
        # target_spans[-1] = tuple(target_spans[-1])
    target.extend(list(chain(*generate_target_spans)))
    target = target[:-1]  # 去掉最后一个ssep
    target.append(1)  # append 1是由于特殊的eos
    return target, target_spans,aspect_label,opinion_label


class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first='A'):
        super(BartBPEABSAPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>',
            "CON": '<<conflict>>',
            "A": '<<aspect:>>',
            "O": '<<opinion:>>',  # value 为加入词表中的值； key 为mapping
            "S":'<<sentiment:>>',
            "SEP":'<<SSEP>>',
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

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
                      '<<X>>',
                      # '<<subword>>',
                      # '<<begin_relation>>',
                      # '<<end_relation>>'

                      ]
        self.tokenizer.add_tokens(pos_tokens)

        # 添加情感信息
        path = './data/senticnet_word.txt'
        self.senticNet = {}
        fp = open(path, 'r')
        for line in fp:
            line = line.strip()
            if not line:
                continue
            word, sentic = line.split('\t')
            self.senticNet[word] = sentic
        fp.close()

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2 +3 # 是由于第一位是sos，紧接着是eos, 然后是+3是指aspect first:     =6+2+3=11

        for name in ['train', 'dev', 'test']:
            ds = data_bundle.get_dataset(name)
            aesc_ds = DataSet()  # 用来做多任务（二元组）的
            if name == 'train':
                trip_ds = aesc_ds  # 用来做三元组的
            else:
                trip_ds = DataSet()
            # a=self.tokenizer.tokenize('aspect',add_prefix_space=True)      #6659
            # b=self.tokenizer.tokenize('opinion',add_prefix_space=True)     #2979
            # b=self.tokenizer.tokenize('pair',add_prefix_space=True)          #1763
            # c=self.tokenizer.tokenize('first:',add_prefix_space=True)      #78，35
            # c=self.tokenizer.tokenize('extract:',add_prefix_space=True)      #14660，35
            # d=self.tokenizer.tokenize('yes,',add_prefix_space=True)      #G,==2156   ,==6
            # e=self.tokenizer.tokenize('polarity:',add_prefix_space=True)      #sentiment,==5702,35  polarity,==8385,21528,35
            # a=self.tokenizer.convert_tokens_to_ids(a)
            # b=self.tokenizer.convert_tokens_to_ids(b)
            # c=self.tokenizer.convert_tokens_to_ids(c)
            # c=self.tokenizer.convert_tokens_to_ids(c)
            # d=self.tokenizer.convert_tokens_to_ids(d)
            # e=self.tokenizer.convert_tokens_to_ids(e)
            # start 准备srctokens
            for ins in ds:
                raw_words = ins['raw_words']
                if 'A' in self.opinion_first:
                    trip_afword_bpes = [[self.tokenizer.bos_token_id], [6659], [78, 35]]    #aspect first:
                    afword_bpes = [[self.tokenizer.bos_token_id], [6659], [14660, 35]]      #aspect extract:
                    ofword_bpes = [[self.tokenizer.bos_token_id],[2979],[14660,35]]         #opinion extract:
                if 'O' in self.opinion_first:
                    trip_ofword_bpes = [[self.tokenizer.bos_token_id], [2979], [78, 35]]    #opinion first:
                word_bpes = [[self.tokenizer.bos_token_id]]

                for word in raw_words:
                    bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                    bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                    if 'A' in self.opinion_first:
                        trip_afword_bpes.append(bpes)
                        afword_bpes.append(bpes)
                        ofword_bpes.append(bpes)
                    if 'O' in self.opinion_first:
                        trip_ofword_bpes.append(bpes)
                    word_bpes.append(bpes)
                if 'A' in self.opinion_first:
                    trip_afword_bpes.append([self.tokenizer.eos_token_id])
                    afword_bpes.append([self.tokenizer.eos_token_id])
                    ofword_bpes.append([self.tokenizer.eos_token_id])
                if 'O' in self.opinion_first:
                    trip_ofword_bpes.append([self.tokenizer.eos_token_id])
                word_bpes.append([self.tokenizer.eos_token_id])

                lens = list(map(len, word_bpes))
                cum_lens = np.cumsum(list(lens)).tolist()
                target = []  # 特殊的开始
                target_spans = []
                generate_target_spans = []
                _word_bpes = list(chain(*word_bpes))
                # end 准备srctokens
                aftarget,aesc_target_spans,aspect_label,opinion_label = get_tgt(ins,'A',cum_lens,_word_bpes,self.tokenizer,target_shift,self.mapping2targetid)
                oftarget, _,_,_ = get_tgt(ins,'O',cum_lens,_word_bpes,self.tokenizer,target_shift,self.mapping2targetid)
                target = [aftarget, oftarget]

                trip_aftarget, trip_target_spans, trip_aspect_label, trip_opinion_label = get_trip_tgt(ins, 'A', cum_lens, _word_bpes,
                                                                              self.tokenizer, target_shift,
                                                                              self.mapping2targetid)
                trip_oftarget, _, _, _ = get_trip_tgt(ins, 'O', cum_lens, _word_bpes, self.tokenizer, target_shift,
                                            self.mapping2targetid)
                trip_target=[trip_aftarget, trip_oftarget]

                '''
                aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
                if self.opinion_first=='O':
                    aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
                elif self.opinion_first=='A':
                    aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))
    
                for aspects, opinions in aspects_opinions:  # 预测bpe的start
                    assert aspects['index'] == opinions['index']
                    a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
                    a_end_bpe = cum_lens[aspects['to']-1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
                    o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
                    o_end_bpe = cum_lens[opinions['to']-1]  # 因为有一个sos shift
                    # 这里需要evaluate是否是对齐的
                    for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                                         (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):
                        assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                               _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]
    
                    if self.opinion_first:
                        generate_target_spans.append(
                            [self.mapping2targetid['O'] + 2, o_start_bpe+target_shift, o_end_bpe+target_shift,
                             self.mapping2targetid['A'] + 2, a_start_bpe+target_shift, a_end_bpe+target_shift ])
                        target_spans.append(
                            [o_start_bpe+target_shift, o_end_bpe+target_shift,
                             a_start_bpe+target_shift, a_end_bpe+target_shift])
                    else:
                        generate_target_spans.append(
                            [self.mapping2targetid['A']+2,a_start_bpe+target_shift, a_end_bpe+target_shift,
                             self.mapping2targetid['O']+2,o_start_bpe+target_shift, o_end_bpe+target_shift,])
                        target_spans.append(
                            [a_start_bpe + target_shift, a_end_bpe + target_shift,
                             o_start_bpe + target_shift, o_end_bpe + target_shift, ])
                    generate_target_spans[-1].extend([self.mapping2targetid['S']+2,self.mapping2targetid[aspects['polarity']]+2])   # 前面有sos和eos
                    generate_target_spans[-1].append(self.mapping2targetid['SEP']+2)
                    generate_target_spans[-1] = tuple(generate_target_spans[-1])
                    target_spans[-1].append(self.mapping2targetid[aspects['polarity']]+2)   # 前面有sos和eos
                    target_spans[-1] = tuple(target_spans[-1])
                target.extend(list(chain(*generate_target_spans)))
                target=target[:-1]  #去掉最后一个ssep
                target.append(1)  # append 1是由于特殊的eos
                '''

                #添加情感信息
                senti_value=[0.0]
                for word in raw_words:
                    bpes_senti = self.tokenizer.tokenize(word, add_prefix_space=True)
                    bpes_senti = self.tokenizer.convert_tokens_to_ids(bpes_senti)
                    if word in self.senticNet:
                        senti_value.extend(len(bpes_senti) * [float(self.senticNet[word])])
                    else:
                        # sentiment.extend(len(bpes)*[0])
                        for s in bpes_senti:
                            if s in self.senticNet:
                                senti_value.append(float(self.senticNet[s]))
                            else:
                                senti_value.append(0.0)
                senti_value.append(0.0)




                import spacy
                spacy.prefer_gpu(0)
                nlp = spacy.load('en_core_web_sm')
                from spacy.tokens import Doc
                # tokenized = self.tokenizer.tokenize(words)   # 不能对句分词，因为上面是对词分词，两种分词结果不一样

                # tokenizedbpes = self.tokenizer.convert_ids_to_tokens(_word_bpes[1:-1])
                # for i in range(len(tokenizedbpes)):
                #     tokenizedbpes[i]=tokenizedbpes[i].replace('Ġ','')
                #     if tokenizedbpes[i]=='':
                #         tokenizedbpes[i]=' '
                pos=Doc(nlp.vocab,
                        # words=tokenizedbpes)
                        words=raw_words)
                for spacyname, tool in nlp.pipeline:
                    tool(pos)
                pos_tag = []
                word_index = []
                itrator=1   # 第一个和最后一个为bos，eos
                for t in pos:
                    # print(t.text, t.i, t.head, t.head.i)
                    conpos='<<'+t.pos_+'>>'              #<<SPACE>>变成unk
                    pos_tag.append(self.tokenizer.convert_tokens_to_ids(conpos))
                    word_index.extend([itrator])
                    if lens[itrator]>1:
                        for i in range(lens[itrator]-1):
                            pos_tag.append(self.tokenizer.convert_tokens_to_ids(conpos))
                            word_index.extend([itrator])
                    itrator += 1
                assert itrator==len(lens)-1
                assert len(pos_tag)==cum_lens[-1]-2
                # ddp = DDParser()
                # 单条句子
                # ddp.parse_seg(tokens)
                # pos_tag = get_tokenized_head(lens[1:-1], pos_tag, cum_lens[-1])

                doc = Doc(nlp.vocab,
                          # words=tokenizedbpes)
                          words=raw_words)
                # Tagger(doc)
                for spacyname, tool in nlp.pipeline:
                    tool(doc)
                head = []
                for t in doc:
                    # print(t.text, t.i, t.head, t.head.i)
                    head.append(t.head.i)
                # ddp = DDParser()
                # 单条句子
                # ddp.parse_seg(tokens)
                head = get_tokenized_head(lens[1:-1], head, cum_lens[-1])

                aesc_ins = Instance(tgt_tokens=target, target_span=aesc_target_spans, src_tokens=_word_bpes.copy(),
                                    afsrc_tokens=list(chain(*afword_bpes)), ofsrc_tokens=list(chain(*ofword_bpes)),
                                    aftgt_tokens=aftarget,oftgt_tokens=oftarget,
                                    head=head, pos_tag=pos_tag,senti_value=senti_value,aspect_label=aspect_label,opinion_label=opinion_label,
                                 word_index=word_index)
                trip_ins = Instance(tgt_tokens=trip_target, target_span=trip_target_spans, src_tokens=_word_bpes.copy(),
                                    afsrc_tokens=list(chain(*trip_afword_bpes)), ofsrc_tokens=list(chain(*trip_ofword_bpes)),
                                    aftgt_tokens=trip_aftarget, oftgt_tokens=trip_oftarget,
                                    head=head, pos_tag=pos_tag, senti_value=senti_value, aspect_label=trip_aspect_label,
                                    opinion_label=trip_opinion_label,
                                    word_index=word_index)
                trip_ds.append(trip_ins)
                aesc_ds.append(aesc_ins)
            if name == 'train':
                data_bundle.set_dataset(trip_ds, 'train')
            else:
                data_bundle.set_dataset(trip_ds, name)

            # return {'tgt_tokens': target, 'target_span': aesc_target_spans, 'src_tokens': list(chain(*word_bpes)),
            #         'afsrc_tokens': list(chain(*afword_bpes)),'ofsrc_tokens': list(chain(*ofword_bpes)),
            #         'aftgt_tokens': aftarget,'oftgt_tokens': oftarget,
            #         'head': head, 'pos_tag':pos_tag, 'senti_value':senti_value, 'word_index':word_index,'aspect_label':aspect_label,'opinion_label':opinion_label}

        # data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('aftgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('oftgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('afsrc_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('ofsrc_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('word_index', -1)

        data_bundle.apply_field(lambda x: len(x), field_name='afsrc_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='aftgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'head', 'pos_tag','senti_value','word_index','aspect_label','opinion_label')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'head', 'pos_tag','senti_value','word_index')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        delete = 0
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            # 删除无a，o的数据
            if len(opinions) == 1:
                if len(opinions[0]['term']) == 0:
                    opinions = []
            if len(aspects) == 1:
                if len(aspects[0]['term']) == 0:
                    aspects = []
            new_aspects = []
            for aspect in aspects:
                # print(aspect)
                # 在有term的aspect中删除的无polarity的数量
                if 'polarity' not in aspect:
                    delete += 1
                    # print('no polarity', delete)
                    continue
                new_aspects.append(aspect)
            ins = Instance(raw_words=tokens, aspects=new_aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds

def get_tokenized_head(lens,head,cumlens):
    assert len(lens)==len(head)
    idx = [i for i in range(len(lens))]
    i=0
    while i < len(lens):
        if lens[i]==2:
            if i !=len(idx)-1:
                idx.append(len(idx))
                head.append(0)
                temp=head[i+1:-1]
                head[i+2:]=temp
                head[i+1]=i             #所有子词的head指向第一个子词
                lens.append(0)
                temlen = lens[i + 1:-1]
                lens[i + 2:] = temlen
                lens[i] = 1
                lens[i + 1] = 1
                for j in range(len(idx)):    #所有指向i后面的head都要加1
                    if head[j]>i:
                        head[j]+=1
            else :
                head.append(i)
                lens[i]=1
                lens.append(1)
        elif lens[i]==3:
            if i !=len(idx)-1:
                idx.append(len(idx))
                idx.append(len(idx))
                head.append(0)
                head.append(0)
                temp=head[i+1:-2]
                head[i+3:]=temp
                head[i+1]=i
                head[i+2]=i
                lens.append(0)
                lens.append(0)
                temlen = lens[i + 1:-2]
                lens[i + 3:] = temlen
                lens[i] = 1
                lens[i + 1] = 1
                lens[i + 2] = 1
                for j in range(len(idx)):
                    if head[j]>i:
                        head[j]+=2
            else :
                head.append(i)
                head.append(i)
                lens[i]=1
                lens.append(1)
                lens.append(1)
        elif lens[i]>3:
            gold_len=lens[i]
            if i !=len(idx)-1:
                for k in range(gold_len-1):
                    idx.append(len(idx))
                    head.append(0)
                    lens.append(0)
                temp=head[i+1:-gold_len+1]
                head[i+gold_len:]=temp
                temlen = lens[i + 1:-gold_len+1]
                lens[i + gold_len:] = temlen
                for k in range(1, gold_len):
                    head[i + k] = i
                    lens[i + k] = 1
                lens[i] = 1
                for j in range(len(idx)):
                    if head[j]>i:
                        head[j]+=gold_len-1
            else :
                lens[i]=1
                for k in range(1, gold_len):
                    head.append(i)
                    lens.append(1)

        i+=1
    assert len(lens)==cumlens-2
    return head

if __name__ == '__main__':
    data_bundle = BartBPEABSAPipe().process_from_file('./../data/pengb/16res')
    print(data_bundle)

