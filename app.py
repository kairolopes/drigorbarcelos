
import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import os

# Carregar o JSON do arquivo 'respostas_drigor.json'
json_path = "respostas_drigor.json"

with open(json_path, "r") as file:
    data = json.load(file)

# Configurar modelo e índice FAISS com as perguntas do JSON
questions = [
    "O que é o Chip da Beleza?",
    "Qual é o endereço da clínica?",
    "Quais são os horários disponíveis para consultas?",
    "Como são realizadas as consultas online?",
    "Qual é o valor da consulta?",
    "Quais são as formas de pagamento aceitas?",
    "Quem é o Dr. Igor Barcelos?"
]

# Mapear perguntas para suas respectivas chaves no JSON
question_to_key = {
    "O que é o Chip da Beleza?": "chip_da_beleza",
    "Qual é o endereço da clínica?": "endereco_clinica",
    "Quais são os horários disponíveis para consultas?": "horarios_consulta",
    "Como são realizadas as consultas online?": "consulta_online",
    "Qual é o valor da consulta?": "valores_consulta",
    "Quais são as formas de pagamento aceitas?": "formas_pagamento",
    "Quem é o Dr. Igor Barcelos?": "quem_e_dr_igor"
}

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Criar a API Flask
app = Flask(__name__)

# Adicionando o endpoint para a raiz
@app.route('/')
def home():
    return jsonify({"message": "Bem-vindo à API do Dr. Igor Barcelos!"}), 200

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "A pergunta está vazia ou não foi enviada"}), 400

    # Codificar a pergunta do usuário e procurar a correspondência mais próxima
    user_embedding = model.encode([user_question])
    _, indices = index.search(user_embedding, k=1)
    matched_question = questions[indices[0][0]]

    # Obter a chave correspondente no JSON e retornar a resposta
    key = question_to_key.get(matched_question)
    if key and key in data:
        response = data[key]["resposta"]
        return jsonify({"response": response})

    return jsonify({"response": "Desculpe, não encontrei a informação solicitada."})

# Adicionar endpoint de verificação de integridade
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
