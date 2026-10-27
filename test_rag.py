import base64
import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer
from unittest.mock import patch
from rag import Index, chunk_documents, cosine, build_prompt, citation_check, generate, embed
from app import Handler, STATE, read_uploads

class PipelineTests(unittest.TestCase):
    def test_tail_overlap_and_page_metadata(self):
        words = ['w'+str(i) for i in range(53)]
        chunks = chunk_documents([{'source':'x.txt','page':4,'text':' '.join(words)}],20,5)
        self.assertEqual(len(chunks),4)
        self.assertEqual(chunks[0].text.split()[-5:],chunks[1].text.split()[:5])
        self.assertEqual(chunks[-1].text.split()[-1],'w52')
        self.assertTrue(all(c.page == 4 for c in chunks))

    def test_invalid_overlap_and_empty(self):
        with self.assertRaises(ValueError): chunk_documents([],20,20)
        with self.assertRaises(ValueError): chunk_documents([{'source':'x','text':'  '}])

    def test_cosine_and_zero(self):
        self.assertAlmostEqual(cosine([1,0],[2,0]),1)
        self.assertEqual(cosine([0,0],[1,2]),0)
        with self.assertRaises(ValueError): cosine([1],[1,2])

    def test_retrieval(self):
        index = Index([{'source':'a','text':'Library books renew fourteen days'},
                       {'source':'b','text':'Laboratory computers software install'}])
        hits = index.search('renew books',1)
        self.assertEqual(hits[0]['source'],'a')
        self.assertTrue(all(h['score']==0 for h in index.search('unicorn galaxy')))

    def test_prompt_has_traceable_evidence(self):
        hits = Index([{'source':'a','text':'Library book loans last fourteen days'}]).search('book')
        prompt = build_prompt('How long?',hits)
        self.assertIn('[S1] File: a; page: 1; chunk: 1',prompt)
        self.assertIn('fourteen days',prompt)
        self.assertEqual(citation_check('Claim [S9]',hits)['unknown'],['S9'])

    def test_embedding_request_contract(self):
        with patch('rag.ollama_post',return_value={'embeddings':[[3,4]]}) as post:
            self.assertEqual(embed(['hello'],True),[[.6,.8]])
            payload = post.call_args.args[1]
            self.assertEqual(payload['input'],['search_query: hello'])
            self.assertFalse(payload['truncate'])

    def test_generation_and_plain_contract(self):
        with patch('rag.ollama_post',return_value={'response':'Answer [S1]'}) as post:
            generate('question',[],use_rag=False)
            self.assertEqual(post.call_args.args[1]['prompt'],'question')
            generate('question',[])
            self.assertIn('EVIDENCE',post.call_args.args[1]['prompt'])
            self.assertFalse(post.call_args.args[1]['stream'])

    def test_upload_decoding(self):
        encoded = base64.b64encode(b'book policy').decode()
        docs, warnings = read_uploads([{'name':'dir/book.txt','data':encoded}])
        self.assertEqual(docs[0]['source'],'book.txt')
        self.assertEqual(docs[0]['text'],'book policy')
        self.assertEqual(warnings,[])
        with self.assertRaises(ValueError): read_uploads([{'name':'x.exe','data':encoded}])

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1',0),Handler)
        cls.url = 'http://127.0.0.1:'+str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self,path,body):
        req=urllib.request.Request(self.url+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r: return json.load(r)

    def test_browser_page_and_index_ask(self):
        with urllib.request.urlopen(self.url) as r:
            self.assertIn(b'Campus Handbook Assistant',r.read())
        result=self.post('/api/index',{'sample':True,'mode':'keyword'})
        self.assertGreater(result['chunks'],0)
        answer=self.post('/api/ask',{'question':'How many books can students borrow?','k':3})
        self.assertIn('library.txt',[h['source'] for h in answer['hits']])
        self.assertEqual(answer['answer_mode'],'evidence')

    def test_missing_index_and_failed_rebuild(self):
        STATE['index']=None
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post('/api/ask',{'question':'test'})
        self.assertEqual(ctx.exception.code,400)
        self.post('/api/index',{'sample':True})
        with self.assertRaises(urllib.error.HTTPError):
            self.post('/api/index',{'sample':True,'size':100,'overlap':100})
        self.assertIsNone(STATE['index'])

    def test_rag_http_with_mocked_model(self):
        with patch('rag.embed',side_effect=lambda texts,query=False:[[1.,0.] for t in texts]), \
             patch('rag.ollama_post',return_value={'response':'Four books [S1].'}):
            self.post('/api/index',{'sample':True,'mode':'semantic'})
            result=self.post('/api/ask',{'question':'How many books?','answer_mode':'rag'})
        self.assertEqual(result['answer'],'Four books [S1].')
        self.assertTrue(result['citations']['has_citation'])

if __name__ == '__main__': unittest.main(verbosity=2)
