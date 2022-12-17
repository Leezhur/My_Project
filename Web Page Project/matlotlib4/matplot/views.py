from django.shortcuts import render

# Create your views here.

from  matplotlib import pyplot as plt
import io
import urllib, base64


import numpy as np
import itertools

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.summarization.summarizer import summarize


    
from django.shortcuts import render, redirect

# Create your views here.
from .forms import BoardForm
from .models import Post, Board

import nltk
nltk.download('punkt')

def token(text):
    
    sentences = sent_tokenize(text)
    vocab = {}
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))

    for sentence in sentences:
        # 단어 토큰화
        tokenized_sentence = word_tokenize(sentence)
        result = []

        for word in tokenized_sentence: 
            word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
            if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
                if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                    result.append(word)
                    if word not in vocab:
                        vocab[word] = 0 
                    vocab[word] += 1
        preprocessed_sentences.append(result) 
        
    vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
    box = []
    for i in vocab_sorted:
        box.append(i)
    box = np.array(box)
    fff = box[0:5, :]
    
    from matplotlib import font_manager, rc
    font_path = "C:/Windows/Fonts/H2GTRM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    x = fff[:, 0]
    y = fff[:, 1]

    x = list(reversed(x))
    y = list(reversed(y))

    plt.plot(x, y, 'ro--')
    plt.bar(x,y) # bar chart 만들기
    plt.title( '시각화' ) # chart의 제목 : '시각화'
    plt.xlabel('단어') # x축 label : '단어'
    plt.ylabel('Score') # y축 label : 'Score'
    plt.grid(True)

    plt.plot(range(10))
    fig = plt.gcf()
    buf = io.BytesIO()

    fig.savefig(buf, format="png")
    buf.seek(0)
    string= base64.b64encode(buf.read())
    data = urllib.parse.quote(string)

    return data

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    word_similarity = cosine_similarity(candidate_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]

    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
        
    return [words[idx] for idx in keywords_idx]

def post(request):
    if request.method == 'POST':
        post = Post()
        post.text = request.POST['text']
        post.save()
        return redirect('post')
    else:
        post = Post.objects.all()
        return render(request, 'post_list.html', {'post':post})
    
def board(request):
    if request.method == 'POST':
        content = request.POST['content']
        
        n_gram_range = (3, 3)
        stop_words = "english"
        
        text = content

        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
        candidates = count.get_feature_names_out()

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([text])
        candidate_embeddings = model.encode(candidates)
        
        a = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.5)

        content = summarize(text, ratio=0.3)

        graph_data = token(text)
        
        board = Board(
            content = content,
            writer = a
        )
        board.save()
        boardForm = BoardForm
        board = Board.objects.all()
        context = {
            'boardForm': boardForm,
            'board': board,
            'data' : graph_data
        }
        return render(request, 'board.html', context)
    else:
        boardForm = BoardForm
        board = Board.objects.all()
        context = {
            'boardForm': boardForm,
            'board': board
        }
        return render(request, 'board.html', context)
    
def aa(context):
    return render(context)
    
def boardEdit(request, pk):
    board = Board.objects.get(id=pk)
    if request.method == "POST":
        board.content = request.POST['content']

        board.save()
        return redirect('board')
    else:
        boardForm = BoardForm(instance=board)
        return render(request, 'update.html', {'boardForm' : boardForm})

def boardDelete(request, pk):
    board = Board.objects.get(id=pk)
    board.delete()
    return redirect('board')