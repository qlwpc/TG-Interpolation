"""Gate full reruns on corrected routing, terminal preservation and GPU batch equivalence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from olmo.config import TrainConfig, ModelConfig
from olmo.data import SentencepieceVocab, get_TG_generate_bias_func
from olmo.eval.downstream import TGPerplexityApproximationDataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);a=parser.parse_args()
    c=a.campaign.resolve();fix=json.loads((c/'grammar_correction.json').read_text());tasks=json.loads((c/'tasks.json').read_text())
    vocab=SentencepieceVocab.from_vocab_file('dataset/bbc-news/TG_GPT2_tokenizer.json')
    data=Path('dataset/bbc-news-reserved-clean-v1/testppl/tree300')
    raw=np.load(data/'tree_300.npy',mmap_mode='r');lengths=np.load(data/'tree_sent_index.npy',mmap_mode='r').reshape(-1)
    raw_sample=np.array(raw[:int(lengths[0])]);report={}
    for mid,record in fix.items():
        cp=Path(record['corrected']);cfg=TrainConfig.load(cp/'config.yaml',validate_paths=False)
        modelcfg=ModelConfig.load(cp/'config.yaml',key='model',validate_paths=False)
        assert cfg.model.transformer_grammar_type==modelcfg.transformer_grammar_type==record['grammar']
        cfg.tokenizer.vocabulary='dataset/bbc-news/TG_GPT2_tokenizer.json'
        assert get_TG_generate_bias_func(cfg) is None
        ds=object.__new__(TGPerplexityApproximationDataset);ds.vocab=vocab;ds.transformer_grammar_type=record['grammar']
        converted=ds._convert_sequence(raw_sample)
        assert not np.array_equal(converted,raw_sample), 'variant conversion did not execute'
        project=lambda x:[int(t) for t in x if vocab.is_terminal(t) or t==vocab.eos]
        assert project(converted)==project(raw_sample),'terminal projection changed'
        task=next(t for t in tasks if t['model']==mid and t['stage']=='pilot');argv=list(task['argv'])
        argv[argv.index('--batch-size')+1]='15';argv[argv.index('--document-ids')+1]='0'
        out=c/'batch15'/mid;argv[argv.index('--output')+1]=str(out)
        subprocess.run([sys.executable,*argv],check=True)
        baseline=json.loads((c/'results'/mid/'document_00000.json').read_text())
        check=json.loads((out/'document_00000.json').read_text())
        assert baseline['fingerprint']==check['fingerprint']
        diffs=[abs(x['nll']-y['nll']) for x,y in zip(baseline['sentences'],check['sentences'])]
        assert len(baseline['sentences'])==len(check['sentences']) and max(diffs)<1e-3
        assert [x['selected_candidate'] for x in baseline['sentences']]==[x['selected_candidate'] for x in check['sentences']]
        assert all((c/'results'/mid/f'document_{i:05d}.json').exists() for i in (0,7,438))
        report[mid]=dict(grammar=record['grammar'],conversion_changes_sequence=True,terminal_projection_preserved=True,max_sentence_nll_batch_difference=max(diffs),selected_history_equal=True)
    (c/'pilot_validation.json').write_text(json.dumps(dict(status='complete',models=report),indent=2)+'\n')

if __name__=='__main__':main()
