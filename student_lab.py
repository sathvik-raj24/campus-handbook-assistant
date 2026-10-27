"""Starter exercises. Read this before opening the solution functions in rag.py.

Run: python student_lab.py
The starter intentionally prints TODO messages until you implement the functions.
After completing it, compare with chunk_documents, cosine, and build_prompt in rag.py.
"""
import math

def split_words(text, size, overlap):
    """Return overlapping word chunks; retain final words; validate overlap < size."""
    # TODO: split(), moving window, break at the end, join().
    raise NotImplementedError('Exercise 1: implement split_words')

def cosine_similarity(a, b):
    """Return dot(a,b)/(norm(a)*norm(b)); return 0 for a zero vector."""
    # TODO: use sum(), zip(), math.sqrt(); reject different dimensions.
    raise NotImplementedError('Exercise 2: implement cosine_similarity')

def evidence_prompt(question, passages):
    """passages is a list of strings. Return a prompt with [S1], [S2], etc."""
    # TODO: number passages, add evidence boundaries, then the question.
    raise NotImplementedError('Exercise 3: implement evidence_prompt')

if __name__ == '__main__':
    exercises = [
        ('Chunking', lambda: split_words('a b c d e f g',4,1) == ['a b c d','d e f g']),
        ('Cosine', lambda: abs(cosine_similarity([1,0],[1,1])-1/math.sqrt(2)) < 1e-9),
        ('Zero vector', lambda: cosine_similarity([0,0],[1,1]) == 0),
        ('Evidence labels', lambda: '[S1]' in evidence_prompt('How long?',['Fourteen days.'])),
    ]
    for title, exercise in exercises:
        try:
            print(title + ': ' + ('PASS' if exercise() else 'TRY AGAIN'))
        except NotImplementedError as exc:
            print(title + ': ' + str(exc))
