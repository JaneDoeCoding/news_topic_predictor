import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(
    __name__,
    template_folder=os.path.join('..', 'frontend', 'templates'),
    static_folder=os.path.join('..', 'frontend', 'static')
)

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# 加载之前保存的embedding数据
data = np.load('news_embeddings.npz', allow_pickle=True)
embeddings = data['embeddings']
texts = data['texts']
categories = data['categories']

def get_user_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def find_most_similar_news(user_input):
    user_vec = get_user_embedding(user_input).reshape(1, -1)
    sims = cosine_similarity(user_vec, embeddings)
    idx = np.argmax(sims)
    return {
        'category': categories[idx].capitalize(),
        'text': texts[idx],
        'similarity_score': float(sims[0][idx])
    }

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收用户新闻标题文本，返回最相似新闻的主题分类，新闻文本摘要，以及相似度分数
    """
    data = request.get_json()
    user_text = data.get('text')
    if not user_text:
        return jsonify({'error': 'No input text provided'}), 400

    result = find_most_similar_news(user_text)

    max_len = 200
    text_snippet = result['text'][:max_len] + ('...' if len(result['text']) > max_len else '')

    response = {
        'category': result['category'].title(),
        'text': text_snippet,
        'similarity_score': round(result['similarity_score'], 3)
    }
    return jsonify(response)

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
