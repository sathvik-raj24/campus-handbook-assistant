"""Retrieval-only evaluation; does not measure answer quality or abstention."""
import argparse
import json
from pathlib import Path
from rag import Index

def evaluate(mode='keyword', size=100, overlap=20, k=3):
    root = Path(__file__).resolve().parent
    docs = [{'source':p.name, 'text':p.read_text(encoding='utf-8')} for p in sorted((root/'data').glob('*.txt'))]
    index = Index(docs, mode, size, overlap)
    questions = json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    supported = [q for q in questions if q['source']]
    source_hits = evidence_hits = 0
    rows = []
    for q in supported:
        hits = index.search(q['question'], k)
        source_hit = any(h['source'] == q['source'] for h in hits)
        retrieved = ' '.join(h['text'].lower() for h in hits if h['source'] == q['source'])
        evidence_hit = all(s.lower() in retrieved for s in q['evidence'])
        source_hits += source_hit
        evidence_hits += evidence_hit
        rows.append({'question':q['question'], 'source_hit':source_hit, 'evidence_hit':evidence_hit})
    return {'mode':mode, 'size_words':size, 'overlap_words':overlap, 'k':k,
            'supported_questions':len(supported), 'source_hit_rate_at_k':source_hits/len(supported),
            'evidence_phrase_coverage_at_k':evidence_hits/len(supported), 'results':rows,
            'note':'Tiny synthetic set. Phrase coverage is a diagnostic, not semantic answer accuracy. '
                   'Manually test the two unsupported questions for abstention and review all generated claims.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['keyword','semantic'], default='keyword')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=20)
    parser.add_argument('--k', type=int, default=3)
    a = parser.parse_args()
    print(json.dumps(evaluate(a.mode, a.size, a.overlap, a.k), indent=2))
