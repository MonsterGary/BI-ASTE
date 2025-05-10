import csv

import torch
from fastNLP.io import Pipe, DataBundle, Loader
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer,BartModel
import numpy as np
from itertools import chain
from functools import cmp_to_key
from collections import defaultdict
import torch.nn.functional as F
from fastNLP import cache_results

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']

def get_tgt(ins,opinion_first,cum_lens,_word_bpes,tokenizer,target_shift,mapping2targetid):
    target = [0]  # 特殊的开始
    target_spans = []
    generate_target_spans = []
    aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
    if opinion_first == 'O':
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))
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
        # assert a_end_bpe>=a_start_bpe
        o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
        o_end_bpe = cum_lens[opinions['to'] - 1]  # 因为有一个sos shift
        if aspects['from']!=-1:
            aspect_[aspects['from']]=2
            if aspects['from']!=aspects['to']-1:
                aspect_[aspects['to']-1]=1
        # else:
        #     aspect_[0]=2
        if opinions['from']!=-1:
            opinion_[opinions['from']] = 2
            if opinions['from']!=opinions['to'] - 1:
                opinion_[opinions['to']-1]=1
        # else:
        #     opinion_[0]=2
        aspect_label.append(aspect_)
        opinion_label.append(opinion_)
        # 这里需要evaluate是否是对齐的

        if opinion_first=='O':
            if aspects['from'] == -1 and opinions['from'] == -1:
                generate_target_spans.append(
                    [mapping2targetid['O'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['A'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
                target_spans.append(
                    [mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
            elif aspects['from'] == -1 and opinions['from'] != -1:
                generate_target_spans.append(
                    [mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift,
                     mapping2targetid['A'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
                target_spans.append(
                    [o_start_bpe + target_shift, o_end_bpe + target_shift,
                     mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
            elif aspects['from'] != -1 and opinions['from'] == -1:
                generate_target_spans.append(
                    [mapping2targetid['O'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift])
                target_spans.append(
                    [mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     a_start_bpe + target_shift, a_end_bpe + target_shift])
            else:
                generate_target_spans.append(
                    [mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift,
                     mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift])
                target_spans.append(
                    [o_start_bpe + target_shift, o_end_bpe + target_shift,
                     a_start_bpe + target_shift, a_end_bpe + target_shift])
        else:
            if aspects['from'] == -1 and opinions['from'] == -1:
                generate_target_spans.append(
                    [mapping2targetid['A'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['O'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
                target_spans.append(
                    [mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2])
            elif aspects['from'] == -1 and opinions['from'] != -1:
                generate_target_spans.append(
                    [mapping2targetid['A'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift, ])
                target_spans.append(
                    [mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2,
                     o_start_bpe + target_shift, o_end_bpe + target_shift, ])
            elif aspects['from'] != -1 and opinions['from'] == -1:
                generate_target_spans.append(
                    [mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift,
                     mapping2targetid['O'] + 2, mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2, ])
                target_spans.append(
                    [a_start_bpe + target_shift, a_end_bpe + target_shift,
                     mapping2targetid['NULL'] + 2, mapping2targetid['NULL'] + 2, ])
            else:
                generate_target_spans.append(
                    [mapping2targetid['A'] + 2, a_start_bpe + target_shift, a_end_bpe + target_shift,
                     mapping2targetid['O'] + 2, o_start_bpe + target_shift, o_end_bpe + target_shift, ])
                target_spans.append(
                    [a_start_bpe + target_shift, a_end_bpe + target_shift,
                     o_start_bpe + target_shift, o_end_bpe + target_shift, ])

        generate_target_spans[-1].extend(
            [mapping2targetid['C'] + 2, mapping2targetid[aspects['category']] + 2,mapping2targetid['S'] + 2, mapping2targetid[aspects['polarity']] + 2])  # 前面有sos和eos
        generate_target_spans[-1].append(mapping2targetid['SEP'] + 2)
        generate_target_spans[-1] = tuple(generate_target_spans[-1])
        target_spans[-1].append(mapping2targetid[aspects['category']] + 2)  # 前面有sos和eos
        target_spans[-1].append(mapping2targetid[aspects['polarity']] + 2)  # 前面有sos和eos
        target_spans[-1] = tuple(target_spans[-1])
    target.extend(list(chain(*generate_target_spans)))
    target = target[:-1]  # 去掉最后一个ssep
    target.append(1)  # append 1是由于特殊的eos
    return target, target_spans,aspect_label,opinion_label



class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first='A', vocab=None, paths=None):
        super(BartBPEABSAPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',  #2
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>',
            "A": '<<aspect:>>',
            "O": '<<opinion:>>',  # value 为加入词表中的值； key 为mapping
            "C": '<<category:>>',
            "S":'<<sentiment:>>',
            "SEP":'<<SSEP>>',
            "NULL": '<<NULL>>',  #10
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
                      '<<commonsense>>',
                      # '<<begin_relation>>',
                      # '<<end_relation>>'

                      ]
        self.tokenizer.add_tokens(pos_tokens)

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
        target_shift = len(self.mapping) + 2 +3 # 是由于第一位是sos，紧接着是eos, 然后是+3是指aspect first:

        import spacy
        spacy.prefer_gpu(0)
        self.spacynlp = spacy.load('en_core_web_sm')
        import stanza
        self.stanzanlp = stanza.Pipeline('en',download_method=None)
        cmargs,cmopt,cmdata_loader,cmmodel,cmtext_encoder=initcomet()




        def prepare_target(ins):
            # a=self.tokenizer.tokenize('aspect',add_prefix_space=True)      #6659
            # b=self.tokenizer.tokenize('opinion',add_prefix_space=True)     #2979
            # c=self.tokenizer.tokenize('first:',add_prefix_space=True)      #78，35
            # d=self.tokenizer.tokenize('yes,',add_prefix_space=True)      #G,==2156   ,==6
            # e=self.tokenizer.tokenize('polarity:',add_prefix_space=True)      #sentiment,==5702,35  polarity,==8385,21528,35
            # a=self.tokenizer.convert_tokens_to_ids(a)
            # b=self.tokenizer.convert_tokens_to_ids(b)
            # c=self.tokenizer.convert_tokens_to_ids(c)
            # d=self.tokenizer.convert_tokens_to_ids(d)
            # e=self.tokenizer.convert_tokens_to_ids(e)
            # start 准备srctokens
            raw_words = ins['words']

            if 'A' in self.opinion_first:
                afword_bpes = [[self.tokenizer.bos_token_id],[6659], [78, 35]]
            if 'O' in self.opinion_first:
                ofword_bpes = [[self.tokenizer.bos_token_id], [2979], [78, 35]]
            word_bpes = [[self.tokenizer.bos_token_id]]

            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                if 'A' in self.opinion_first:
                    afword_bpes.append(bpes)
                if 'O' in self.opinion_first:
                    ofword_bpes.append(bpes)
                word_bpes.append(bpes)
            # if 'A' in self.opinion_first:
            #     afword_bpes.append([self.tokenizer.eos_token_id])
            # if 'O' in self.opinion_first:
            #     ofword_bpes.append([self.tokenizer.eos_token_id])
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            generate_target_spans = []
            _word_bpes = list(chain(*word_bpes))
            # end 准备srctokens
            aftarget, target_spans, afaspect_label, afopinion_label = get_tgt(ins, 'A', cum_lens, _word_bpes,
                                                                              self.tokenizer, target_shift,
                                                                              self.mapping2targetid)
            oftarget, _, ofaspect_label, ofopinion_label = get_tgt(ins, 'O', cum_lens, _word_bpes, self.tokenizer,
                                                                   target_shift, self.mapping2targetid)
            target = [aftarget, oftarget]
            aspect_label = [afaspect_label, ofaspect_label]
            if afaspect_label != ofaspect_label:
                print(aspect_label)



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


            # POS SPACY
            # HEAD SPACY
            from spacy.tokens import Doc
            spacydoc = Doc(self.spacynlp.vocab,
                      # words=tokenizedbpes)
                      words=raw_words)
            for spacyname, tool in self.spacynlp.pipeline:
                tool(spacydoc)
            # pos_tag = []
            # postag=[]
            pos_mask=[]
            head = []
            deprel = []
            itrator=1   # 第一个和最后一个为bos，eos
            for t in spacydoc:
                #     # print(t.text, t.i, t.head, t.head.i)
                # head.append(t.head.i)  # 第一个单词为0，从0开始数
                # deprel.append(t.dep_)
                conpos='<<'+t.tag_+'>>'              #<<SPACE>>变成unk
                # pos_tag.append(self.tokenizer.convert_tokens_to_ids(conpos))
                # postag.append(t.tag_)
                pos_mask.extend([itrator])
                if lens[itrator]>1:
                    for i in range(lens[itrator]-1):
                        pos_mask.extend([itrator])
                itrator += 1


            # STANZA POS
            stanzadoc = self.stanzanlp(ins['raw_words'])
            postag = []
            pos_tag = []
            #head = []
            #deprel = []
            for sentence in stanzadoc.sentences:
                for word in sentence.words:
                    #pos_tag.append(word.xpos)
                    conpos = '<<' + word.upos+ '>>'  # <<SPACE>>变成unk
                    pos_tag.append(self.tokenizer.convert_tokens_to_ids(conpos))
                    head.append(word.head)
                    deprel.append(word.deprel)


            #postag = list(ins['postag'])
            #pos_tag = [self.postag_vocab.stoi.get(t,self.postag_vocab.unk_index) for t in postag]
            #if self.postag_vocab.unk_index in pos_tag:
            #   print(postag)
            #head=ins['head']
            #deprel=ins['deprel']
            deprel_ids=[self.deprel_vocab.stoi.get(t,self.deprel_vocab.unk_index) for t in deprel]
            max_sequence_len=100
            #if max_sequence_len<=len(pos_tag):
            #    print(len(pos_tag))

            """2. generate deprel index of the word pair"""
            word_pair_deprel = torch.zeros(max_sequence_len, max_sequence_len).long()
            matrx_mask = torch.zeros(max_sequence_len,max_sequence_len).long()
            for i in range(len(head)):
                    if head[i] == 0:
                        word_pair_deprel[i + 1][i + 1] = self.deprel_vocab.stoi.get('root')
                        matrx_mask[i + 1][i + 1] = 1
                        continue
                    word_pair_deprel[i+1][head[i]] = self.deprel_vocab.stoi.get(deprel[i],self.deprel_vocab.unk_index)
                    #if self.deprel_vocab.stoi.get(deprel[i],self.deprel_vocab.unk_index)==self.deprel_vocab.unk_index:
                    #    print(deprel[i])
                    word_pair_deprel[head[i]][i+1] = self.deprel_vocab.stoi.get(deprel[i],self.deprel_vocab.unk_index)
                    word_pair_deprel[i+1][i+1] = self.deprel_vocab.stoi.get('<self>')
                    matrx_mask[i+1][head[i]] = 1
                    matrx_mask[head[i]][i+1] = 1
                    matrx_mask[i+1][i+1] = 1
            '''1. generate position index of the word pair'''
            word_pair_position = torch.zeros(max_sequence_len, max_sequence_len).long()
            for i in range(len(head)):
                for j in range(len(head)):
                    word_pair_position[i+1][j+1] = self.post_vocab.stoi.get(abs(i - j), self.post_vocab.unk_index)

            tree_based_word_pair_position = torch.zeros(max_sequence_len, max_sequence_len).long()
            tmp = [[0] * len(head) for _ in range(len(head))]
            for i in range(len(head)):
                j = head[i]
                if j == 0:
                    continue
                tmp[i][j - 1] = 1
                tmp[j - 1][i] = 1

            tmp_dict = defaultdict(list)
            for i in range(len(head)):
                for j in range(len(head)):
                    if tmp[i][j] == 1:
                        tmp_dict[i].append(j)

            word_level_degree = [[4] * len(head) for _ in range(len(head))]

            for i in range(len(head)):
                node_set = set()
                word_level_degree[i][i] = 0
                node_set.add(i)
                for j in tmp_dict[i]:
                    if j not in node_set:
                        word_level_degree[i][j] = 1
                        node_set.add(j)
                    for k in tmp_dict[j]:
                        if k not in node_set:
                            word_level_degree[i][k] = 2
                            node_set.add(k)
                            for g in tmp_dict[k]:
                                if g not in node_set:
                                    word_level_degree[i][g] = 3
                                    node_set.add(g)
            for i in range(len(head)):
                for j in range(len(head)):
                    tree_based_word_pair_position[i+1][j+1] = self.synpost_vocab.stoi.get(word_level_degree[i][j], self.synpost_vocab.unk_index)

            # COMET inference
            appendix = '<<commonsense>> '

            cmoutput = cometinference(cmargs, cmopt, cmdata_loader, cmmodel, cmtext_encoder,
                                      "PersonX say " + ins['raw_words'], "xReact", "topk-1")
            if cmoutput['xReact']['beams'][0] != 'none' and cmoutput['xReact']['beams'][0] != '':
                appendix = appendix + "I feel " + cmoutput['xReact']['beams'][0] + '.'

            appendix=appendix+'Given example: '+ins['example']['sentence']+" target: "+str(ins['example']['target'])
            sents=[]
            if appendix != '':
                line = appendix.split('\t')
                words=line[0]
                tuples=line[1:]
                tuples=converttuples2json(tuples)
                sents.append(words.split())
            for word in sents[0]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                if 'A' in self.opinion_first:
                    afword_bpes.append(bpes)
                if 'O' in self.opinion_first:
                    ofword_bpes.append(bpes)

            if 'A' in self.opinion_first:
                afword_bpes.append([self.tokenizer.eos_token_id])
            if 'O' in self.opinion_first:
                ofword_bpes.append([self.tokenizer.eos_token_id])

            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes)),
                    'afsrc_tokens': list(chain(*afword_bpes)), 'ofsrc_tokens': list(chain(*ofword_bpes)),
                    'aftgt_tokens': aftarget, 'oftgt_tokens': oftarget,
                    'head': head,'word_pair_deprel':word_pair_deprel,'matrix_mask':matrx_mask, 'pos_tag':pos_tag,
                    'senti_value':senti_value, 'word_index':pos_mask,'deprel_ids':deprel_ids,'word_pair_position':word_pair_position,
                    'tree_based_word_pair_position':tree_based_word_pair_position,
                    'afaspect_label': afaspect_label, 'afopinion_label': afopinion_label,
                    'ofaspect_label': ofaspect_label, 'ofopinion_label': ofopinion_label,
                    }

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('aftgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('oftgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('word_index', -1)
        data_bundle.set_pad_val('afsrc_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('ofsrc_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='afsrc_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='aftgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='pos_tag', new_field_name='head_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'head',
                              'word_pair_deprel','matrix_mask', 'pos_tag','senti_value','word_index','head_len','deprel_ids',
                              'word_pair_position','tree_based_word_pair_position',
                              'afaspect_label', 'afopinion_label',
                              'ofaspect_label', 'ofopinion_label',
                              )
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'head',
                               'word_pair_deprel','matrix_mask', 'pos_tag','senti_value','word_index','head_len','deprel_ids',
                               'word_pair_position','tree_based_word_pair_position',)

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
        ds = DataSet()
        data, categories,original_line = get_transformed_io(path)
        sorted_line = []
        for ins in data:
            raw_words = ins['raw_words']
            words = ins['sentence']
            Triples = ins['triples']
            term = ins['term']
            aspects = []
            opinions = []
            i = 0
            delete_flag=0
            for triple in Triples:
                aspect = {'index': -1, 'from': -1, 'to': -1, 'polarity': -1, 'term': -1, 'category':-1}
                opinion = {'index': -1, 'from': -1, 'to': -1, 'term': -1}
                aspect['index'] = i
                opinion['index'] = i
                aspect['from'] = triple[0][0]
                aspect['to'] = triple[0][1]
                aspect['term'] = term[i][0]
                aspect['polarity'] = triple[3][:3].upper()
                aspect['category'] = triple[2]
                opinion['from'] = triple[1][0]
                opinion['to'] = triple[1][1]
                opinion['term'] = term[i][1]
                aspects.append(aspect)
                opinions.append(opinion)
                i += 1
                if (aspect['from']==-1 or opinion['from']==-1):
                    delete_flag=1
            if delete_flag==0:
                sorted_line.append(original_line[ins['id']])

            if len(aspects) != len(opinions):
                print('wrong in ABSALoader')
            # postag=ins['postag']
            # head=ins['head']             #root=0, 第一个单词为1, 从1开始数。
            # deprel=ins['deprel']
            ins = Instance(raw_words=raw_words,words=words, aspects=aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds) > 30:
                break

        return ds


def get_spans(tags,words):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i])
                start = -1
    if start != -1:
        spans.append([start, length])
    return spans,words[spans[0][0]:spans[0][1]]
def get_transformed_io(data_path):
    """
    The main function to transform the Input & Output according to
    the specified paradigm and task
    """
    raw_words, sents, labels, original_line = read_line_examples_from_file(data_path)
    with open(r'E:\PycharmWorkingSpace\BARTABSA-Syntactic\BARTABSA-main\data\acos\res2acos\dev.txt', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for i in range(len(raw_words)):
            if len(labels[i])==0:
                continue
            sorted_line = []
            sorted_line.append(raw_words[i]+'####')
            # for t in labels[i]:
                # quad=
            sorted_line.append(labels[i])
            writer.writerow(sorted_line)
            print(sorted_line)
    return 0

sentnum2word = {'2': 'POS', '0': 'NEG', '1': 'NEU'}

def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels, raw_words, original_line = [], [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            original_line.append(line)
            line = line.strip()
            if line != '':
                line = line.split('\t')
                words=line[0]
                tuples=line[1:]
                tuples=converttuples2json(tuples)
                sents.append(words.split())
                labels.append(eval(tuples))
                raw_words.append(words)
    print(f"Total examples = {len(sents)}")
    return raw_words, sents, labels, original_line

def converttuples2json(tuples):  #
    preds=[]
    for ele_ori in tuples:
        ele=ele_ori.split()
        # asp=eval(ele[0])
        # if -1 in asp:
        #     continue
        # opi=eval(ele[3])
        # if -1 not in opi:
        #     continue
        senti=sentnum2word[ele[2]]
        # senti=ele[2]
        # category='<<'+ele[1].lower()+'>>'
        category=ele[1].lower().replace('#', ' ').replace('_', ' ')
        # 使用 split() 方法将字符串分割成列表
        resulta = ele[0].split(',')

        # 将字符串列表转换为整数列表
        asp = [int(i) for i in resulta]
        # 使用 split() 方法将字符串分割成列表
        resulto = ele[3].split(',')

        # 将字符串列表转换为整数列表
        opi = [int(i) for i in resulto]
        pred = (asp,opi,senti,category)
        preds.append(pred)
    return str(preds)

if __name__ == '__main__':
    data_bundle = get_transformed_io(r'E:\PycharmWorkingSpace\BARTABSA-Syntactic\BARTABSA-main\data\acos\Restaurant-ACOS\rest16_quad_dev.tsv')
    print(data_bundle)

