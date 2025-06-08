import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

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

def predict_topic(user_input):
    user_vec = get_user_embedding(user_input).reshape(1, -1)
    sims = cosine_similarity(user_vec, embeddings)
    idx = np.argmax(sims)
    
    category = categories[idx].capitalize()
    snippet = texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx]
    score = float(sims[0][idx])
    
    return category, snippet, round(score, 3)

# 创建 Gradio 接口
demo = gr.Interface(
    fn=predict_topic,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline"),
    outputs=[
        gr.Label(label="Predicted Topic Category"),
        gr.Textbox(label="Most Similar News Article Snippet"),
        gr.Label(label="Similarity Score")
    ],
    title="News Topic Predictor",
    description="Paste a news headline to see its topic and the most similar news article from our database."
)

demo.launch()
