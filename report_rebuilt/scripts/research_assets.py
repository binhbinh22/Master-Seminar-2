"""EDA, learning diagnostics and illustrative annotations from existing artifacts."""
from pathlib import Path
import json, sys
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'report_rebuilt/scripts'))
from audit import tex,table
OUT=ROOT/'report_rebuilt'
LANGS=['python','java','pharo']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
data={l:pd.read_csv(ROOT/'data/k_fold_data'/l/'all.csv') for l in LANGS}
labels={l:json.loads((ROOT/'data/k_fold_data'/l/'label_cols.json').read_text()) for l in LANGS}
def save(name,fig):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight',dpi=220)
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharex=True)
for ax,l in zip(axes,LANGS):
    s=data[l][labels[l]].mean().sort_values()
    ax.barh(s.index,100*s,color='#4285a1');ax.set_title(l.title());ax.set_xlabel('Positive prevalence (%)');ax.set_xlim(0,48)
    for i,v in enumerate(s):ax.text(100*v+.5,i,f'{100*v:.1f}',va='center',fontsize=10)
fig.tight_layout();save('eda_labels',fig)

stats={};fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for l in LANGS:
    words=data[l].text.str.split().str.len().to_numpy(); chars=data[l].text.str.len().to_numpy()
    stats[l]={'word_median':float(np.median(words)),'word_p95':float(np.percentile(words,95)),'word_max':int(words.max()),'char_median':float(np.median(chars))}
    for ax,arr in zip(axes,[words,chars]):
        a=np.sort(arr);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=l.title())
for ax,title in zip(axes,['Whitespace-delimited words','Characters']):
    ax.set_xscale('log');ax.set_xlabel(title+' (log scale)');ax.set_ylabel('Cumulative sentence fraction');ax.grid(alpha=.2)
axes[0].legend();fig.tight_layout();save('eda_lengths',fig)

fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for ax,l in zip(axes,LANGS):
    a=data[l][labels[l]].to_numpy(dtype='int64');co=a.T@a;support=a.sum(0)
    conditional=co/support[:,None]
    im=ax.imshow(conditional,vmin=0,vmax=1,cmap='Blues')
    short=[s if len(s)<16 else s[:13]+'...' for s in labels[l]]
    ax.set_xticks(range(len(short)),short,rotation=65,ha='right',fontsize=9);ax.set_yticks(range(len(short)),short,fontsize=9);ax.set_title(l.title())
fig.subplots_adjust(wspace=.9,bottom=.34,left=.12,right=.93,top=.9)
fig.colorbar(im,ax=axes,shrink=.7,label='P(column label | row label)',pad=.02)
save('eda_cooccurrence',fig)

fig,axes=plt.subplots(2,3,figsize=(11,6),sharex='col');curve_stats={}
for j,l in enumerate(LANGS):
    p=next((ROOT/'results/training/final').glob(f'*/ovr_logreg/{l}/training_curve.json'))
    curve=json.loads(p.read_text());x=[r['step'] for r in curve]
    axes[0,j].plot(x,[r['train_loss'] for r in curve],color='#4285a1',marker='.',label='Train Hamming loss')
    axes[1,j].plot(x,[r['validation_micro_f1'] for r in curve],label='Validation micro-F1')
    axes[1,j].plot(x,[r['validation_macro_f1'] for r in curve],label='Validation macro-F1')
    axes[0,j].set_title(l.title());axes[1,j].set_xlabel('Maximum L-BFGS iterations')
    for ax in axes[:,j]:ax.set_xscale('log');ax.grid(alpha=.2)
    curve_stats[l]={'source':p.relative_to(ROOT).as_posix(),'first_train_hl':curve[0]['train_loss'],'last_train_hl':curve[-1]['train_loss'],'last_val_micro':curve[-1]['validation_micro_f1']}
axes[0,0].set_ylabel('Training Hamming loss');axes[1,0].set_ylabel('Validation F1');axes[1,0].legend(fontsize=9)
fig.tight_layout();save('ml_learning',fig)

fig,axes=plt.subplots(2,3,figsize=(11,6),sharex='col');neural=[]
for j,l in enumerate(LANGS):
    for exp,col in [('exp1','#4285a1'),('exp15','#bb713b')]:
        p=next((ROOT/'results/training').glob(f'final_gpu*/{exp}/{l}/metrics.json'));d=json.loads(p.read_text());h=d['training']['history'];x=[r['epoch'] for r in h]
        name='E1' if exp=='exp1' else 'E15'
        axes[0,j].plot(x,[r['train_loss'] for r in h],label=name,color=col)
        axes[1,j].plot(x,[r['validation']['micro_f1'] for r in h],label=name,color=col,marker='.')
        b=d['training']['best_epoch'];best=next(r for r in h if r['epoch']==b)
        axes[1,j].scatter([b],[best['validation']['micro_f1']],color=col,marker='*',s=110,zorder=4)
        neural.append({'language':l,'model':name,'best_epoch':b,'completed_epochs':len(h),'source':p.relative_to(ROOT).as_posix()})
    axes[0,j].set_title(l.title());axes[1,j].set_xlabel('Epoch')
    for ax in axes[:,j]:ax.grid(alpha=.2)
axes[0,0].set_ylabel('Training objective');axes[1,0].set_ylabel('Validation micro-F1');axes[0,0].legend()
fig.tight_layout();save('neural_learning',fig)

def escape(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in s)
examples=[]
for l,target in [('python','Parameters'),('python','Usage'),('java','rational'),('pharo','Classreferences')]:
    df=data[l];subset=df[(df[target]==1)&(df.text.str.len().between(35,145))]
    if l=='python' and target=='Usage':subset=subset[subset[labels[l]].sum(axis=1)>1]
    row=subset.sort_values('comment_sentence_id').iloc[0]
    examples.append({'language':l,'id':str(row.comment_sentence_id),'text':row.text,'labels':[k for k in labels[l] if row[k]==1]})
content=r'\begin{table}[htbp]\centering\small'+'\n'+r'\caption{Actual annotated sentences illustrating information types. These are dataset examples, not model errors or predictions.}\label{tab:examples}'+'\n'+r'\begin{tabularx}{\linewidth}{llXX}\toprule Language & ID & Sentence & Annotated labels\\\midrule'+'\n'
for e in examples:content+=' & '.join([e['language'].title(),e['id'],escape(e['text']),', '.join(escape(k) for k in e['labels'])])+r'\\'+'\n'
content+=r'\bottomrule\end{tabularx}\end{table}'+'\n'
(OUT/'tables/illustrations.tex').write_text(content,encoding='utf-8')
def cv_value(exp,language):
    p=next((ROOT/'results/training').glob(f'kfold_gpu*/{exp}/{language}/metrics.json'))
    return Decimal(f'{json.loads(p.read_text())["cross_validation"]["micro_f1"]["mean"]:.4f}')
deltas={'pharo_E3_minus_E1_pp':str(100*(cv_value('exp3','pharo')-cv_value('exp1','pharo'))),
        'pharo_E15_minus_E11_pp':str(100*(cv_value('exp15','pharo')-cv_value('exp11','pharo')))}
assert Decimal(deltas['pharo_E3_minus_E1_pp'])==Decimal('1.08')
assert Decimal(deltas['pharo_E15_minus_E11_pp'])==Decimal('1.51')
rational={}
for exp in ['exp1','exp17']:
    p=next((ROOT/'results/training').glob(f'final_gpu*/{exp}/java/metrics.json'))
    v=json.loads(p.read_text())['test']['per_label']['rational']
    rational[exp]={'tp':v['true_positive'],'fp':v['predicted_positive']-v['true_positive'],'fn':v['support']-v['true_positive'],'support':v['support']}
assert rational=={'exp1':{'tp':3,'fp':2,'fn':25,'support':28},'exp17':{'tp':12,'fp':2,'fn':16,'support':28}}
(OUT/'research_assets.json').write_text(json.dumps({'lengths':stats,'ml_curves':curve_stats,'neural_curves':neural,'examples':examples,'displayed_score_deltas':deltas,'java_rational':rational},indent=2),encoding='utf-8')
print(json.dumps({'lengths':stats,'ml_curves':curve_stats,'examples':examples},indent=2))
