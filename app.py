import os
import json
from flask import Flask, request, render_template, jsonify
from transformers import pipeline

app = Flask(__name__)

# Carrega o banco de conhecimento
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)
    intents = data['intents']

labels = [intent['tag'] for intent in intents]

# Inicializa o pipeline zero-shot com o modelo BART large MNLI
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Rota home (página inicial)
@app.route('/')
def home():
    return render_template('index.html')

# Rota para responder às perguntas
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message'].lower()

    # Usa o zero-shot para classificar a intenção do usuário
    result = classifier(user_input, labels)

    # Pega a intent mais provável e sua confiança
    predicted_intent = result['labels'][0]
    confidence = result['scores'][0]

    # Define um limite mínimo de confiança para responder
    threshold = 0.15

    # Resposta manual para saudações simples
    greetings = ['olá', 'oi', 'ola', 'bom dia', 'boa tarde', 'e aí', 'fala']
    if any(greet == user_input for greet in greetings):
        return jsonify({"response": "Olá! Como posso ajudar você hoje?"})

    if confidence >= threshold:
        # Busca a resposta da intent prevista
        for intent in intents:
            if intent['tag'] == predicted_intent:
                response = intent['resposta'][0]
                break
    else:
        response = "Desculpe, não entendi. Pode reformular?"

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
