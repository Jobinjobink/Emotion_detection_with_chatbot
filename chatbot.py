import random

emotion_responses = {
    "Happy": [
        "You seem really happy today 😊 What's making you smile?",
        "That's great to see! Want to share something positive?"
    ],
    "Sad": [
        "I'm here for you 💙 Do you want to talk about what's bothering you?",
        "It's okay to feel sad sometimes. I'm listening."
    ],
    "Angry": [
        "Let's take a deep breath together 😌 What's going on?",
        "I understand. Want to talk it out calmly?"
    ],
    "Fear": [
        "You're safe here 🤍 What's worrying you?",
        "I'm here to help you feel more comfortable."
    ],
    "Surprise": [
        "That sounds unexpected 😲 Want to tell me more?",
        "Wow! That must have been surprising."
    ],
    "Disgust": [
        "I understand. Let's talk about it calmly.",
        "That doesn't sound pleasant. Want to explain?"
    ],
    "Neutral": [
        "How can I help you today?",
        "I'm here. What would you like to talk about?"
    ]
}

def chatbot_response(user_input, emotion):
    base_responses = emotion_responses.get(emotion, emotion_responses["Neutral"])
    response = random.choice(base_responses)

    # simple continuation
    if "hello" in user_input.lower():
        response += " Hello! 👋"

    return response
