import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer
        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1, encoding='utf-8')
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({ 'pred': pred, 'label': label }, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0: report[k] = 0
            else:
                if 'dist' in k: v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = { 'bleu@1': 0, 'bleu@2': 0, 'bleu@3': 0, 'bleu@4': 0,
                        'dist@1': set(), 'dist@2': set(), 'dist@3': set(), 'dist@4': set(), }
        self.sent_cnt = 0
    
    def after_eval_report(self, preds, labels):
        self.collect_ngram(preds)
        self.compute_bleu(preds, labels)
        self.sent_cnt += len([pred for pred in preds if len(pred) > 0])
        pass

def conv_gen_eval():
    evaluator = ConvEvaluator(None, None)
    rep_path = "/home/work/CRSTEST/KERS_HJ/epoch_output/2/2023-07-23_052257_BKERS_3711Train_3711Test_1e-5_facebook_bart-base/12_test_GEN_REPORT.txt"
    preds, labels = [],[]
    with open(rep_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            linedic = json.loads(line)
            preds.append(linedic['pred'])
            labels.append(linedic['label'])
    evaluator.after_eval_report(preds, labels)
    report = evaluator.report()
    report_text = [f"bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                  f"{report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
    print(report_text[0],'\n',report_text[1])


if __name__=='__main__':
    import json
    conv_gen_eval()