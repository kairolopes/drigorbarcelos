import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

# Carregar o JSON do arquivo 'respostas_drigor.json'
json_path = "respostas_drigor.json"
with open(json_path, "r") as file:
    data = json.load(file)

# Configurar modelo e índice FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preparar perguntas e respostas do JSON
questions = []
answers = []
for key, value in data.items():
    questions.append(value['pergunta'])
    answers.append(value['resposta'])

# Criar embeddings para as perguntas
embeddings = model.encode(questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Criar a API Flask
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Bem-vindo à API do Dr. Igor Barcelos!"}), 200

@app.route('/get_answer', methods=['POST'])
def get_answer():
    # Log para depuração - ver o que o ChatVolts está enviando
    data = request.json
    print("Dados recebidos do ChatVolts:", data)

    # Verificar se a chave "question" está presente
    user_question = data.get("question")
    if not user_question:
        return jsonify({"error": "A pergunta está vazia ou não foi enviada"}), 400

    # Codificar a pergunta do usuário
    user_embedding = model.encode([user_question])

    # Procurar a pergunta mais próxima usando FAISS
    _, indices = index.search(user_embedding, k=1)
    matched_index = indices[0][0]

    # Buscar a resposta correspondente
    if matched_index < len(answers):
        response = answers[matched_index]
        return jsonify({"response": response})

    return jsonify({"response": "Desculpe, não encontrei a informação solicitada."}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
