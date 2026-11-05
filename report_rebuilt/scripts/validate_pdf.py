"""Inspect compiled PDF structure, bounds, log, and render every page for review."""
from pathlib import Path
import json,re,unicodedata
import pymupdf as fitz
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/rendered';OUT.mkdir(parents=True,exist_ok=True)
doc=fitz.open(ROOT/'main.pdf')
cover=re.sub(r'\s+','',unicodedata.normalize('NFC',doc[0].get_text()))
cover_fields=['HANOI UNIVERSITY OF SCIENCE AND TECHNOLOGY','SCHOOL OF INFORMATION & COMMUNICATION TECHNOLOGY',
              'Dr. Phuong Nguyen','Nguyễn Khắc Thái Bình','20251324M','Lã Minh Trung','20251319M',
              'Nguyễn Thu Uyên','20252279M','Nguyễn Thị Nhã Linh','20261262M','Nguyễn Đức Anh','20252469M',
              'Course: Seminar 2','Academic Year: 2026']
for field in cover_fields:
    assert re.sub(r'\s+','',unicodedata.normalize('NFC',field)) in cover,field
log=(ROOT/'main.log').read_text(errors='replace')
errors=[line for line in log.splitlines() if re.search(r'Overfull|LaTeX Warning|Package .* Warning|undefined|Missing character|^!',line)]
assert not errors,errors
edges=[]
for i,p in enumerate(doc):
    assert p.get_text().strip(),f'Blank page {i+1}'
    for block in p.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<25 or x1>p.rect.width-25 or y0<20 or y1>p.rect.height-20:
                    edges.append({'page':i+1,'text':span['text'],'bbox':span['bbox']})
assert not edges,edges
for start in range(0,len(doc),6):
    sheet=Image.new('RGB',(1000,2160),'#dce2e6');draw=ImageDraw.Draw(sheet)
    for k in range(6):
        n=start+k
        if n>=len(doc):break
        pix=doc[n].get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((480,680))
        x=k%2*500+10;y=k//2*720+25
        sheet.paste(im,(x,y));draw.text((x,y-18),f'PDF page {n+1}',fill='black')
    sheet.save(OUT/f'contact_{start//6+1}.png')
for i,p in enumerate(doc):
    text=p.get_text()
    if any(term in text for term in ['Module-level architecture','Implemented experimental pipeline','Actual two-stage protocol','Figure 6:','Python test F1 by label','Java test F1 by label']):
        p.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(OUT/f'latest_page_{i+1:02}.png')
result={'pages':len(doc),'figures':len(list((ROOT/'figures').glob('*.pdf'))),'generated_table_files':len(list((ROOT/'tables').glob('*.tex'))),
        'latex_warnings_or_overflow':errors,'out_of_bounds_text':edges,'rendered_pages':len(doc),'bibtex_succeeded':(ROOT/'main.bbl').exists(),
        'cover_metadata_verified':True,'compiler':'XeLaTeX + BibTeX'}
result['bibliography_entries']=len(re.findall(r'\\bibitem\{',(ROOT/'main.bbl').read_text(encoding='utf-8')))
result['numbered_tables']=len(re.findall(r'\\contentsline\s*\{table\}',(ROOT/'main.lot').read_text(encoding='utf-8')))
(ROOT/'pdf_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
