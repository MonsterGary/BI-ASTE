from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter


class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, opinion_first=True):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos

        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.quad_fp = 0
        self.quad_tp = 0
        self.quad_fn = 0
        self.em = 0
        self.invalid = 0
        self.total = 0
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        # assert opinion_first is False, "Current metric only supports aspect first"

        self.opinin_first = opinion_first

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 0, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_span,invalid = acos_parse(pred,pred_seq_len,self.word_start_index,target_span)
        self.invalid=invalid
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred_span)):
            '''
            em = 0     # 精确匹配率，预测序列中的每个元素都与目标序列中的对应元素相等，那么em的值将为1，表示完全匹配；
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index and j not in(5,6,7,8):
                        cur_pair.append(j)
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    elif j not in(5,6,7,8):
                        cur_pair.append(j)
                    elif j in(5,6,7,8):
                        continue
            pred_spans.append(pairs.copy())
            self.invalid += invalid
            '''
            pairs=ps
            oe_ae_target = [tuple(t[:4]) for t in ts]
            oe_ae_pred = [p[:4] for p in pairs]

            oe_ae_target_counter = Counter(oe_ae_target)
            oe_ae_pred_counter = Counter(oe_ae_pred)
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_ae_pred_counter.keys())),
                                           set(list(oe_ae_target_counter.keys())))
            self.ae_oe_fn += fn
            self.ae_oe_fp += fp
            self.ae_oe_tp += tp

            # note aesc

            if self.opinin_first=='O':    #t=o o a a c s
                ae_sc_target = [(t[2], t[3], t[-1]) for t in ts]
                ae_sc_pred = [(p[2], p[3], p[-1]) for p in pairs]
            else:
                ae_sc_target = [(t[0], t[1], t[-1]) for t in ts]
                ae_sc_pred = [(p[0], p[1], p[-1]) for p in pairs]
            asts = set([tuple(t) for t in ae_sc_target])
            asps = set(ae_sc_pred)
            for p in list(asps):  # pairs is a 5-tuple
                if p in asts:
                    asts.remove(p)
                    self.ae_sc_tp += 1
                else:
                    self.ae_sc_fp += 1
            self.ae_sc_fn += len(asts)

            # note triplet

            aste_target = [(t[0], t[1],t[2], t[3], t[-1]) for t in ts]
            aste_pred = [(p[0], p[1],p[2], p[3], p[-1]) for p in pairs]
            tripts = set([tuple(t) for t in aste_target])
            tripps = set(aste_pred)
            for p in list(tripps):
                if p in tripts:
                    tripts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1
            self.triple_fn += len(tripts)

            # note ACOS

            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.quad_tp += 1
                else:
                    self.quad_fp += 1
            self.quad_fn += len(ts)

    def get_metric(self, reset=True):
        res = {}

        f, pre, rec = _compute_f_pre_rec(1, self.quad_tp, self.quad_fn, self.quad_fp)
        # rec：所有样本中预测出来的样本
        # pre：所有预测中预测出来的样本

        res['quad_f'] = round(f, 4) * 100
        res['quad_rec'] = round(rec, 4) * 100
        res['quad_pre'] = round(pre, 4) * 100


        f, pre, rec = _compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)
        # rec：所有样本中预测出来的样本
        # pre：所有预测中预测出来的样本

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_oe_tp, self.ae_oe_fn, self.ae_oe_fp)

        res['oe_ae_f'] = round(f, 4)*100
        res['oe_ae_rec'] = round(rec, 4)*100
        res['oe_ae_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_sc_tp, self.ae_sc_fn, self.ae_sc_fp)
        res["ae_sc_f"] = round(f, 4)*100
        res["ae_sc_rec"] = round(rec, 4)*100
        res["ae_sc_pre"] = round(pre, 4)*100

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.ae_oe_fp = 0
            self.ae_oe_tp = 0
            self.ae_oe_fn = 0
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.quad_fp = 0
            self.quad_tp = 0
            self.quad_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 0
            self.ae_sc_fp = 0
            self.ae_sc_tp = 0
            self.ae_sc_fn = 0
        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, set):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, set):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)        # 该groundtruth在预测结果里，tp=1. else 0
        fp += max(p_num - t_num, 0)    # 该groundtruth不在预测结果里，fp=1，else 0
        fn += max(t_num - p_num, 0)    # 该groundtruth在预测结果里，fn=0，else 1
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp


def acos_parse(pred, pred_seq_len, word_start_index,target):
    pred_span=[]
    invalid = 0
    pred_nums = len(pred)//2
    for i, ps in enumerate(pred.tolist()):
        ps = ps[:pred_seq_len[i]]
        pairs = []
        cur_pair = []
        if len(ps):
            for index, j in enumerate(ps):
                if j < 11 and j not in (5, 6, 7, 8,9,10):
                    cur_pair.append(j)
                    if len(cur_pair) != 6 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                        invalid += 1
                    else:
                        if i>=pred_nums:
                        #     t=cur_pair[:2].copy()
                        #     cur_pair[:2]=cur_pair[2:4].copy()
                        #     cur_pair[2:4]=t
                            pairs.append((cur_pair[2], cur_pair[3], cur_pair[0], cur_pair[1],cur_pair[4],cur_pair[5]))
                        else:
                            pairs.append(tuple(cur_pair))
                    cur_pair = []
                elif j not in (5, 6, 7, 8,9):
                    cur_pair.append(j)
                elif j in (5, 6, 7, 8, 9):
                    continue
        pred_span.append(pairs.copy())
    pred_span = distance_aware_merge_preds(pred_span,target)
    return pred_span, invalid

def distance_aware_merge_preds(preds,targets):
    pred_nums = len(preds) // 2
    merged_preds = []
    for i in range(pred_nums):
        ao_pred, oa_pred = preds[i], preds[i + pred_nums]
        aotar = targets[i]
        if ao_pred == oa_pred:
            pred = ao_pred
        else:
            ao_pred = list([x for x in ao_pred])
            oa_pred = list([x for x in oa_pred])
            pred = []
            ao_pred_dup, oa_pred_dup = ao_pred.copy(), oa_pred.copy()
            # add the common triplet into ans (intersection set)
            for cur in ao_pred:
                if cur in oa_pred_dup:
                    ao_pred_dup.remove(cur)
                    oa_pred_dup.remove(cur)
                    pred.append(cur)
        merged_preds.append(pred)
    return merged_preds