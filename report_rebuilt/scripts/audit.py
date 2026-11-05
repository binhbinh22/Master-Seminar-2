"""Read-only evidence audit and deterministic report tables/figures. No training."""
from pathlib import Path
import json, csv, hashlib, subprocess, statistics, sys
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'report_rebuilt'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')
def table(name,headers,rows,caption,label,align=None):
    content='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
    content+='\\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}\\toprule\n'
    content+=' & '.join(headers)+r'\\\midrule'+'\n'
    content+='\n'.join(' & '.join(str(x) for x in row)+r'\\' for row in rows)
    content+='\n\\bottomrule\\end{tabular}\n\\end{table}\n'
    (OUT/'tables'/f'{name}.tex').write_text(content,encoding='utf-8')
def figsave(name):
    plt.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.savefig(OUT/'figures'/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

records=[]
for p in sorted((ROOT/'results/training').rglob('metrics.json')):
    d=read(p)
    if d.get('stage') in ('kfold','final'):
        records.append((p,d))
facts={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'python_audit':sys.version,'datasets':{},'runs':[],'summary_mismatches':[]}
datasetrows=[]; labelrows=[]
for lang in ['python','java','pharo']:
    folder=ROOT/'data/k_fold_data'/lang
    df=pd.read_csv(folder/'all.csv',dtype={'comment_sentence_id':str})
    labels=read(folder/'label_cols.json'); folds=read(folder/'folds.json')
    raw=pd.read_csv(ROOT/'data'/f'{lang}.csv',dtype={'comment_sentence_id':str})
    wide=raw.pivot(index='comment_sentence_id',columns='category',values='instance_type')
    assert len(df)==len(wide)
    assert np.array_equal(df[labels].to_numpy(),wide.loc[df.comment_sentence_id,labels].to_numpy())
    valid=[]
    for f in folds:
        tr=df.iloc[f['train_indices']]; va=df.iloc[f['val_indices']]
        assert not set(tr.group_id)&set(va.group_id)
        valid.extend(f['val_indices'])
    assert sorted(valid)==list(range(len(df)))
    splits={s:pd.read_csv(ROOT/'data/full_training_data'/lang/f'{s}.csv',dtype={'comment_sentence_id':str}) for s in ['train','val','test']}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(splits[a].group_id)&set(splits[b].group_id)
    joined=pd.concat(splits.values()).set_index('comment_sentence_id')
    assert np.array_equal(df[labels].to_numpy(),joined.loc[df.comment_sentence_id,labels].to_numpy())
    positives=int(df[labels].sum().sum()); negatives=len(df)*len(labels)-positives
    info={'sentences':len(df),'labels':labels,'positive':positives,'negative':negatives,
          'cardinality':positives/len(df),'density':positives/(len(df)*len(labels)),
          'unique_text_groups':int(df.group_id.nunique()),'duplicate_excess':len(df)-int(df.group_id.nunique()),
          'multilabel_rows':int((df[labels].sum(axis=1)>1).sum()),'zero_label_rows':int((df[labels].sum(axis=1)==0).sum()),
          'split_counts':{s:len(v) for s,v in splits.items()},
          'class_overlap_train_test':len(set(splits['train']['class'])&set(splits['test']['class'])),
          'test_classes':int(splits['test']['class'].nunique()),
          'supports':{l:int(df[l].sum()) for l in labels},
          'test_supports':{l:int(splits['test'][l].sum()) for l in labels}}
    facts['datasets'][lang]=info
    datasetrows.append([lang.title(),len(df),len(labels),positives,negatives,f'{info["cardinality"]:.3f}',info['unique_text_groups']])
    for l in labels: labelrows.append([lang.title(),tex(l),int(df[l].sum()),f'{100*df[l].mean():.2f}',int(splits['train'][l].sum()),int(splits['val'][l].sum()),int(splits['test'][l].sum())])
table('dataset',['Language','$N$','$L$','Positive','Negative','Cardinality','Groups'],datasetrows,'Verified sentence and target counts. Groups identify exact whitespace-normalized text.','tab:data')
table('labels',['Language','Label','Total +','Prevalence (\\%)','Train +','Val +','Test +'],labelrows,'Positive support by label and Stage 2 split; prevalence uses the full language population.','tab:labels','llrrrrr')

cv={}; final={}
for p,d in records:
    method=d.get('model',d.get('method')); lang=d['language']; key=(lang,method)
    target=cv if d['stage']=='kfold' else final
    if key in target: raise ValueError(f'Duplicate run {key}')
    target[key]=(p,d)
    metrics=d['cross_validation'] if d['stage']=='kfold' else d['test']
    if d['stage']=='kfold':
        for m,summary in metrics.items():
            vals=[f.get('validation',f)[m] for f in d['folds'] if f.get('validation',f).get(m) is not None]
            assert abs(statistics.mean(vals)-summary['mean'])<1e-10
            assert abs(statistics.stdev(vals)-summary['std'])<1e-10
    else:
        per=metrics['per_label']; tp=sum(x['true_positive'] for x in per.values())
        pred=sum(x['predicted_positive'] for x in per.values()); sup=sum(x['support'] for x in per.values())
        assert abs(2*tp/(pred+sup)-metrics['micro_f1'])<1e-10
        assert abs(statistics.mean(x['f1'] for x in per.values())-metrics['macro_f1'])<1e-10
    facts['runs'].append({'path':p.relative_to(ROOT).as_posix(),'language':lang,'model':method,'stage':d['stage'],'seed':d['seed'],
                          'data_fingerprint':d['data_fingerprint'],'population_fingerprint':d['population_fingerprint']})
for lang in facts['datasets']:
    for stage,collection in [('kfold',cv),('final',final)]:
        runs=[d for (l,m),(p,d) in collection.items() if l==lang]
        assert len({d['data_fingerprint'] for d in runs})==1
        assert len({d['population_fingerprint'] for d in runs})==1

names={'ovr_logreg':'Logistic regression','classifier_chain_logreg':'Classifier chain','ovr_linear_svc':'Linear SVM','ovr_sgd':'SGD','ovr_random_forest':'Random forest','dummy_most_frequent':'Majority dummy'}
ml=list(names)
rows=[]
for method in ml+['E1']:
    rows.append([names.get(method,'RoBERTa E1')]+[f'{cv[(l,method)][1]["cross_validation"]["micro_f1"]["mean"]:.4f} $\\pm$ {cv[(l,method)][1]["cross_validation"]["micro_f1"]["std"]:.4f}' for l in facts['datasets']])
table('ml_cv',['Method','Python','Java','Pharo'],rows,'Stage 1 micro-F1: mean and sample standard deviation across five folds. E1 uses fold validation for checkpoint selection.','tab:mlcv','lrrr')
rows=[]
for i in range(1,18):
    if i==4: continue  # E4 has exactly E1 screening scores; explain once in the caption.
    model=f'E{i}'; rows.append([model]+[f'{cv[(l,model)][1]["cross_validation"]["micro_f1"]["mean"]:.4f} $\\pm$ {cv[(l,model)][1]["cross_validation"]["micro_f1"]["std"]:.4f}' for l in facts['datasets']])
table('transformer_cv',['Configuration','Python','Java','Pharo'],rows,'Transformer screening mean and sample SD. E4 shares E1 scores in all three languages and is omitted because threshold tuning is deferred. All 51 run records remain available.','tab:dlcv','lrrr')
for lang in facts['datasets']:
    for stage,collection in [('cv',cv),('test',final)]:
        rows=[]
        for (l,m),(p,d) in collection.items():
            if l!=lang: continue
            if stage=='cv' and m=='E4': continue
            vals=d['cross_validation'] if stage=='cv' else d['test']
            get=lambda key: vals[key]['mean'] if stage=='cv' else vals[key]
            rows.append([tex(names.get(m,m))]+[f'{get(key):.4f}' for key in ['micro_f1','macro_f1','hamming_loss','micro_average_precision']])
        rows.sort(key=lambda row:-float(row[1]))
        note=' E4 shares E1 screening scores and is omitted.' if stage=='cv' else ''
        table(f'{lang}_{stage}',['Method','Micro-F1','Macro-F1','HL','Micro-AP'],rows,f'{lang.title()} {"Stage 1 fold means" if stage=="cv" else "Stage 2 test results"}. HL is Hamming loss (lower is better).'+note,f'tab:{lang}{stage}','lrrrr')

for lang in facts['datasets']:
    model=max([(m,d['cross_validation']['micro_f1']['mean']) for (l,m),(p,d) in cv.items() if l==lang and m.startswith('E')],key=lambda x:x[1])[0]
    facts['datasets'][lang]['cv_leader']=model
    present=[(m,d['test']['micro_f1']) for (l,m),(p,d) in final.items() if l==lang and m.startswith('E')]
    facts['datasets'][lang]['test_dl_leader']=max(present,key=lambda x:x[1])[0]

fig,axs=plt.subplots(1,3,figsize=(11,3.1),sharey=True)
for ax,lang in zip(axs,facts['datasets']):
    methods=ml[0:5]+['E1']; vals=[cv[(lang,m)][1]['cross_validation']['micro_f1'] for m in methods]
    ax.bar(range(6),[v['mean'] for v in vals],yerr=[v['std'] for v in vals],capsize=2,color=['#4285a1']*5+['#bb713b'])
    ax.set_xticks(range(6),['LR','Chain','SVM','SGD','RF','E1'],rotation=35); ax.set_title(lang.title()); ax.set_ylim(0,.9); ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Stage 1 micro-F1'); fig.tight_layout(); figsave('ml_vs_e1')
fig,axs=plt.subplots(1,3,figsize=(11,3.1),sharey=True)
for ax,lang in zip(axs,facts['datasets']):
    models=['E1','E3','E9','E15','E17']; x=np.arange(5)
    ax.bar(x-.18,[cv[(lang,m)][1]['cross_validation']['micro_f1']['mean'] for m in models],.36,label='CV mean',color='#4285a1')
    ax.bar(x+.18,[final[(lang,m)][1]['test']['micro_f1'] for m in models],.36,label='Exploratory test',color='#bb713b')
    ax.set_xticks(x,models); ax.set_title(lang.title()); ax.set_ylim(0,.85); ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Micro-F1'); fig.legend(*axs[-1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.12),ncol=2,fontsize=12,frameon=False); fig.tight_layout(); figsave('cv_test')
for lang in facts['datasets']:
    models=['E1','E3','E9','E15','E17']; labels=facts['datasets'][lang]['labels']
    matrix=np.array([[final[(lang,m)][1]['test']['per_label'][l]['f1'] for l in labels] for m in models])
    fig,ax=plt.subplots(figsize=(8,2.8)); im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_yticks(range(5),models); ax.set_xticks(range(len(labels)),labels,rotation=23,ha='right')
    for i in range(5):
        for j in range(len(labels)): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>.55 else '#1d3445',fontsize=10)
    fig.colorbar(im,ax=ax,label='Test F1',pad=.02); fig.tight_layout(); figsave(f'{lang}_heatmap')
    rows=[]
    for label in labels:
        base=final[(lang,'E1')][1]['test']['per_label'][label]
        best=final[(lang,facts['datasets'][lang]['test_dl_leader'])][1]['test']['per_label'][label]
        row=[tex(label),base['support'],f'{base["precision"]:.3f}',f'{base["recall"]:.3f}',f'{base["f1"]:.3f}']
        if facts['datasets'][lang]['test_dl_leader']!='E1':
            row += [f'{best["precision"]:.3f}',f'{best["recall"]:.3f}',f'{best["f1"]:.3f}']
        rows.append(row)
    leader=facts['datasets'][lang]['test_dl_leader']
    headers=['Label','Support','E1 P','E1 R','E1 F1']
    if leader!='E1': headers += [f'{leader} P',f'{leader} R',f'{leader} F1']
    table(f'{lang}_perlabel',headers,rows,f'{lang.title()} test label performance. The highest observed test micro-F1 Transformer is {leader}; comparator choice is retrospective and descriptive.',f'tab:{lang}labels','l'+'r'*(len(headers)-1))

configs=[]
base=yaml.safe_load((ROOT/'configs/exp1_roberta_baseline.yaml').read_text())
for i in range(1,18):
    p=next((ROOT/'configs').glob(f'exp{i}_*.yaml')); d=yaml.safe_load(p.read_text())
    changes={k:v for k,v in d['model'].items() if k not in ['name','hidden_size','threshold']}
    changes.update({k:v for k,v in d['training'].items() if base['training'].get(k)!=v})
    configs.append({'experiment':f'E{i}','path':p.relative_to(ROOT).as_posix(),'backbone':d['model']['name'],'variant':d['variant'],'changes':changes})
facts['configs']=configs
for stage,summary in [('kfold','results/stage1_summary/stage1_metrics.csv'),('final','results/stage2_summary/stage2_metrics.csv')]:
    df=pd.read_csv(ROOT/summary)
    for _,r in df.iterrows():
        d=(cv if stage=='kfold' else final)[(r.language,r.model)][1]
        key='micro_f1_mean' if stage=='kfold' else 'test_micro_f1'
        actual=d['cross_validation']['micro_f1']['mean'] if stage=='kfold' else d['test']['micro_f1']
        if abs(r[key]-actual)>1e-10: facts['summary_mismatches'].append([summary,r.language,r.model,key])
facts['prediction_files']=[p.relative_to(ROOT).as_posix() for p in (ROOT/'results').rglob('*predictions*') if p.is_file()]
facts['model_checkpoints']=[p.relative_to(ROOT).as_posix() for p in (ROOT/'results').rglob('best_model.pt')]
(OUT/'audit_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
sources=[]
for pattern in ['src/**/*.py','experiments/**/*.py','configs/**/*.yaml','data/**/*.csv','data/**/*.json','results/**/metrics.json','results/*_summary/*.csv','docs/**/*.md','docs/**/*.tex','notebooks/*.ipynb','reports/*.pdf','requirements.txt','README.md','scripts/*.sh']:
    for p in ROOT.glob(pattern): sources.append(p)
with (OUT/'source_inventory.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['path','sha256','bytes'])
    for p in sorted(set(sources)): w.writerow([p.relative_to(ROOT).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size])
print(json.dumps({'datasets':facts['datasets'],'runs':len(records),'predictions':facts['prediction_files'],'mismatches':facts['summary_mismatches'],'configs':configs},indent=2))
