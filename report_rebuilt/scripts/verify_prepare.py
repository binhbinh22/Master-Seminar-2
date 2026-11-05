"""Recreate staged datasets in temporary storage; never modify archived evidence."""
from pathlib import Path
import json, tempfile, os, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.prepare_two_stage import prepare
from src.data.two_stage import load_stage_data

os.chdir(ROOT)
checks=[]
with tempfile.TemporaryDirectory(prefix='report_split_check_') as tmp:
    counts=prepare('data',tmp)
    for lang in ['python','java','pharo']:
        for folder,stage in [('k_fold_data','kfold'),('full_training_data','final')]:
            stored=load_stage_data(ROOT/'data'/folder,lang,stage)
            generated=load_stage_data(Path(tmp)/folder,lang,stage)
            assert stored.population_fingerprint==generated.population_fingerprint
            assert stored.data_fingerprint==generated.data_fingerprint
            assert stored.sentence_ids==generated.sentence_ids
            assert (stored.labels==generated.labels).all()
            assert stored.folds==generated.folds
            checks.append({'language':lang,'stage':stage,'data_fingerprint':stored.data_fingerprint,'passed':True})
out={'counts':counts,'checks':checks,'research_training_rerun':False}
(ROOT/'report_rebuilt/split_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
