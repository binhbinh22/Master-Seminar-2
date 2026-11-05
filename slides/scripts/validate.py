from pathlib import Path
import re,json,zipfile
import pymupdf as fitz
from PIL import Image,ImageOps,ImageDraw
root=Path(__file__).resolve().parents[1]
out=root/'build/rendered';out.mkdir(parents=True,exist_ok=True)
doc=fitz.open(root/'main.pdf')
log=(root/'main.log').read_text(errors='replace')
issues=[line for line in log.splitlines() if re.search(r'Overfull|LaTeX Warning|Package .* Warning|Missing character|^!',line)]
bounds=[]
for i,page in enumerate(doc):
    assert page.get_text().strip(),i+1
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0:continue
        for line in block['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:bounds.append((i+1,span['text']))
    page.get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(out/f'slide_{i+1:02}.png')
for start in range(0,len(doc),6):
    sheet=Image.new('RGB',(1200,1080),'#e5e9ed');draw=ImageDraw.Draw(sheet)
    for j in range(start,min(start+6,len(doc))):
        im=Image.open(out/f'slide_{j+1:02}.png');im.thumbnail((590,330))
        x=(j-start)%2*600;y=(j-start)//2*360
        draw.text((x+8,y+3),f'Slide {j+1}',fill='black');sheet.paste(im,(x+5,y+23))
    sheet.save(out/f'contact_{start//6+1}.png')
result={'slides':len(doc),'aspect_ratio':'16:9','latex_issues':issues,'text_outside_slide':bounds,'rendered_slides':len(doc),'theme_commit':'15102a28bdd3ec1fde97538bb7567e277862ba1c'}
(root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
assert len(doc)==25 and not issues and not bounds
