import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

# Carregar o JSON do arquivo 'respostas.json'
json_path = "respostas.json"
try:
    with open(json_path, "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Erro: Arquivo '{json_path}' não encontrado.")
    data = []
except json.JSONDecodeError:
    print(f"Erro: Arquivo '{json_path}' está corrompido ou mal formatado.")
    data = []

# Configurar modelo e índice FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preparar perguntas e respostas do JSON
questions = []
answers = []

# Iterar corretamente sobre os itens da lista
for entry in data:
    # Verificar se as chaves 'pergunta' e 'resposta' existem no JSON
    pergunta = entry.get("pergunta")
    resposta = entry.get("resposta")
    if pergunta and resposta:
        questions.append(pergunta)
        answers.append(resposta)
    else:
        print(f"Entrada inválida no JSON: {entry}")

# Criar embeddings para as perguntas (somente se houver perguntas válidas)
if questions:
    embeddings = model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
else:
    print("Nenhuma pergunta válida encontrada no JSON.")
    index = None

# Criar a API Flask
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Bem-vindo à API do Dr. Igor Barcelos!"}), 200

@app.route('/get_answer', methods=['POST'])
def get_answer():
    # Log para depuração - ver os dados recebidos
    req_data = request.json
    print("Dados recebidos:", req_data)

    # Verificar se a chave "pergunta" está presente
    user_question = req_data.get("pergunta")
    if not user_question:
        return jsonify({"error": "A pergunta está vazia ou não foi enviada."}), 400

    # Verificar se o índice FAISS foi criado
    if index is None:
        return jsonify({"error": "O sistema não está pronto. Nenhuma pergunta foi carregada."}), 500

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
