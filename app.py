"""Local single-user teaching server. Run: python app.py"""
import argparse
import base64
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from rag import Index, build_prompt, generate, citation_check

ROOT = Path(__file__).resolve().parent
STATE = {'index': None}

def summarize_text(text, limit=360):
    """Fast extractive overview: preserve the first useful sentences from each source."""
    cleaned = ' '.join(text.split())
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', cleaned) if part.strip()]
    summary = ' '.join(sentences[:3]) or cleaned
    if len(summary) > limit:
        summary = summary[:limit].rsplit(' ', 1)[0].rstrip('.,;:') + '…'
    return summary or 'No readable text found.'

def read_uploads(files):
    docs, warnings = [], []
    for f in files:
        source = str(f['name']).replace('\\', '/').split('/')[-1][:120]
        raw = base64.b64decode(f['data'], validate=True)
        suffix = Path(source).suffix.lower()
        if suffix in ('.txt', '.md'):
            docs.append({'source': source, 'page': 1, 'text': raw.decode('utf-8-sig')})
        elif suffix == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError('PDF needs pypdf: python -m pip install -r requirements.txt') from exc
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError('Use an unencrypted PDF.')
            if len(reader.pages) > 50:
                raise ValueError('Workshop limit: 50 pages per PDF.')
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ''
                if text.strip():
                    docs.append({'source': source, 'page': number, 'text': text})
                else:
                    warnings.append('%s page %s has no text; OCR is not included.' % (source, number))
        else:
            raise ValueError('Only UTF-8 TXT, MD, or text-based PDF files are supported.')
    if sum(len(d['text']) for d in docs) > 500000:
        raise ValueError('Workshop limit: 500,000 extracted characters.')
    return docs, warnings

class Handler(BaseHTTPRequestHandler):
    def send(self, data, status=200, content_type='application/json'):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/':
            return self.send((ROOT/'index.html').read_bytes(), content_type='text/html')
        if self.path == '/api/status':
            idx = STATE['index']
            return self.send({'indexed': idx is not None, 'chunks': len(idx.chunks) if idx else 0})
        self.send({'error': 'Not found'}, 404)

    def do_POST(self):
        # Same-origin JSON requests only; server is bound to loopback.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'Cross-origin requests are not allowed.'}, 403)
        if self.headers.get_content_type() != 'application/json':
            return self.send({'error': 'Use application/json.'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8_000_000:
                raise ValueError('Request must be under 8 MB.')
            body = json.loads(self.rfile.read(length))
            start = time.perf_counter()
            if self.path == '/api/index':
                STATE['index'] = None
                if body.get('sample', False):
                    documents = [{'source': p.name, 'page': 1, 'text': p.read_text(encoding='utf-8')}
                                 for p in sorted((ROOT/'data').glob('*.txt'))]
                    warnings = []
                else:
                    documents, warnings = read_uploads(body.get('files', []))
                # Failed rebuilds invalidate old state, avoiding answers from stale documents.
                idx = Index(documents, body.get('mode', 'keyword'),
                            int(body.get('size', 100)), int(body.get('overlap', 20)))
                STATE['index'] = idx
                grouped = {}
                for doc in documents:
                    entry = grouped.setdefault(doc['source'], {'pages': set(), 'text': []})
                    entry['pages'].add(int(doc.get('page', 1)))
                    entry['text'].append(doc.get('text', ''))
                chunk_counts = Counter(chunk.source for chunk in idx.chunks)
                doc_details = [{'source': source, 'pages': len(entry['pages']),
                                'chunks': chunk_counts.get(source, 0),
                                'summary': summarize_text(' '.join(entry['text']))}
                               for source, entry in grouped.items()]
                result = {'chunks': len(idx.chunks), 'mode': idx.mode, 'warnings': warnings,
                          'documents': doc_details,
                          'preview': [vars(c) for c in idx.chunks[:5]]}
            elif self.path == '/api/ask':
                idx = STATE['index']
                question = str(body.get('question', '')).strip()
                if not question or len(question) > 1500:
                    raise ValueError('Question must contain 1..1500 characters.')
                history = body.get('history', [])
                if not isinstance(history, list) or len(history) > 6:
                    raise ValueError('Conversation history must contain at most 6 turns.')
                safe_history = []
                for turn in history:
                    if not isinstance(turn, dict):
                        raise ValueError('Invalid conversation turn.')
                    user_text, assistant_text = str(turn.get('user', ''))[:1500], str(turn.get('assistant', ''))[:3000]
                    if user_text and assistant_text:
                        safe_history.append({'user': user_text, 'assistant': assistant_text})
                mode = body.get('answer_mode', 'evidence')
                if mode not in ('evidence', 'rag', 'plain'):
                    raise ValueError('Unknown answer mode.')
                # Handle standalone greetings conversationally, before retrieval or
                # constructing a document-grounded prompt.
                greeting = re.fullmatch(
                    r"(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening)|"
                    r"how\s+are\s+you|how\s+is\s+it\s+going|what['’]?s\s+up|sup|greetings)[\s.!?,]*",
                    question.casefold())
                if mode == 'rag' and greeting:
                    answer = 'Hello! I’m doing well, thanks for asking. What would you like to know about your documents?'
                    result = {'answer': answer, 'hits': [], 'answer_mode': mode,
                              'prompt': '', 'citations': None}
                    result['seconds'] = round(time.perf_counter()-start, 3)
                    return self.send(result)
                if idx is None:
                    raise ValueError('Build an index first.')
                retrieval_query = ' '.join([t['user'] for t in safe_history[-3:]] + [question])
                hits = idx.search(retrieval_query, int(body.get('k', 3)))
                # Zero keyword overlap is a useful baseline rejection; semantic scores have no universal cutoff.
                no_overlap = idx.mode == 'keyword' and hits[0]['score'] <= 0
                if mode == 'evidence':
                    answer = ('No matching words. I do not know from these documents.' if no_overlap
                              else 'Retrieved passages below. This mode does not generate an answer.')
                elif no_overlap and mode == 'rag':
                    answer = "I don't know from these documents. No keyword overlap was found."
                else:
                    model = body.get('model', 'qwen2.5:1.5b')
                    if model not in ('qwen2.5:1.5b', 'qwen2.5:0.5b'):
                        raise ValueError('Choose one of the provided local generation models.')
                    answer = generate(question, hits, model, use_rag=mode == 'rag', history=safe_history)
                result = {'answer': answer, 'hits': hits, 'answer_mode': mode,
                          'prompt': question if mode == 'plain' else build_prompt(question, hits, safe_history),
                          'citations': citation_check(answer, hits) if mode == 'rag' else None}
            else:
                return self.send({'error': 'Not found'}, 404)
            result['seconds'] = round(time.perf_counter()-start, 3)
            self.send(result)
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            self.send({'error': str(exc)}, 400)
        except Exception as exc:
            self.send({'error': str(exc)}, 502)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print('Campus Handbook Assistant: http://127.0.0.1:%s' % args.port)
    print('Single-user classroom demo. Ctrl+C to stop. Documents stay in memory.')
    try:
        HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
