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

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Simulação de respostas do chatbot
chatbot_responses = {
    "imunidade": "Alimentos ricos em vitamina C e zinco são ideais.",
    "proteína vegetal": "Feijão, lentilha, soja e grão-de-bico são boas fontes de proteína vegetal.",
    "cultivo sustentável": "Use compostagem, rotação de culturas e reaproveitamento de água.",
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message'].lower()
    response = "Desculpe, não entendi. Pode reformular?"

    for intent in intents:
        for exemplo in intent['exemplos']:
            if exemplo in user_input:
                response = intent['resposta'][0]
                break
        else:
            continue
        break

    return jsonify({"response": response})
    

if __name__ == '__main__':
    app.run(debug=True)