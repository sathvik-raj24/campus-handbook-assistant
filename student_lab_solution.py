"""Reference answers for student_lab.py. Open after attempting the exercises."""
import math

def split_words(text, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Size must be positive; overlap must be 0 <= overlap < size.')
    words = text.split()
    result = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        result.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return result

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError('Vectors need the same dimensions.')
    dot = sum(x*y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x*x for x in a))
    length_b = math.sqrt(sum(y*y for y in b))
    if length_a == 0 or length_b == 0:
        return 0.0
    return dot/(length_a*length_b)

def evidence_prompt(question, passages):
    evidence = '\n\n'.join('[S%s] %s' % (i, text)
                            for i, text in enumerate(passages, 1))
    return ('Use only the evidence below. Cite each fact with its source label. '
            'If the answer is absent, say "I don\'t know from these documents."\n'
            'EVIDENCE\n' + evidence + '\nEND EVIDENCE\nQUESTION\n' + question)

if __name__ == '__main__':
    print(split_words('a b c d e f g',4,1))
    print(round(cosine_similarity([1,0],[1,1]),6))
    print(evidence_prompt('How long?',['Book loans last fourteen days.']))
