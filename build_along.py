"""Run each pipeline stage separately, after studying its implementation in rag.py.

Examples:
    python build_along.py --stage load
    python build_along.py --stage chunks
    python build_along.py --stage search --mode keyword
    python build_along.py --stage prompt
    python build_along.py --stage answer
    python build_along.py --stage answer --mode semantic
"""
import argparse
import json
from pathlib import Path
from rag import Index, chunk_documents, build_prompt, generate, citation_check

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['load','chunks','search','prompt','answer'],default='load')
    parser.add_argument('--mode',choices=['keyword','semantic'],default='keyword')
    parser.add_argument('--size',type=int,default=100)
    parser.add_argument('--overlap',type=int,default=20)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--question',default='How many books can a third year student borrow, and for how long?')
    args = parser.parse_args()
    root=Path(__file__).resolve().parent
    docs=[{'source':p.name,'page':1,'text':p.read_text(encoding='utf-8')}
          for p in sorted((root/'data').glob('*.txt'))]
    if args.stage == 'load':
        print(json.dumps(docs,indent=2))
        return
    chunks=chunk_documents(docs,args.size,args.overlap)
    if args.stage == 'chunks':
        print(json.dumps([vars(c) for c in chunks],indent=2))
        return
    index=Index(docs,args.mode,args.size,args.overlap)
    hits=index.search(args.question,args.k)
    if args.stage == 'search':
        print(json.dumps(hits,indent=2))
        return
    prompt=build_prompt(args.question,hits)
    if args.stage == 'prompt':
        print(prompt)
        return
    answer=generate(args.question,hits)
    print(answer)
    print('\nCitation label check (not factual verification):')
    print(json.dumps(citation_check(answer,hits),indent=2))

if __name__ == '__main__':
    main()
