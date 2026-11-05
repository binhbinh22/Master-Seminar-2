"""Create the Overleaf source ZIP; optionally compile an extracted copy."""
from pathlib import Path
import argparse, subprocess, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify',action='store_true')
args=parser.parse_args()
files=[ROOT/'main.tex',ROOT/'main.pdf',ROOT/'references.bib',ROOT/'source_inventory.csv']
files += [p for name in ['sections','tables','figures','scripts'] for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += list(ROOT.glob('*.md'))+list(ROOT.glob('*.json'))
target=ROOT/'report_rebuilt_overleaf.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)): archive.write(path,path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    if args.verify:
        with tempfile.TemporaryDirectory(prefix='seminar_overleaf_') as folder:
            archive.extractall(folder)
            for command in [['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],
                            ['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
                if command[0]=='xelatex': command.insert(1,'--disable-installer')
                result=subprocess.run(command,cwd=folder,capture_output=True,text=True,errors='replace',timeout=180)
                if result.returncode: raise RuntimeError(result.stdout[-3000:]+result.stderr[-1000:])
            import pymupdf
            expected=len(pymupdf.open(ROOT/'main.pdf'))
            actual=len(pymupdf.open(Path(folder)/'main.pdf'))
            assert actual==expected,(actual,expected)
            print(f'Extracted source bundle compiled successfully: {actual} pages.')
print(f'Created {target.name}: {len(set(files))} files, {target.stat().st_size:,} bytes.')
