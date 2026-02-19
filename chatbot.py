import json
import random
import re

import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from textblob import TextBlob  # type: ignore


class NMCCChatbot:
    courses = {
        "UG_Aided": [
            "B.A. Tamil Literature",
            "B.A. English Literature",
            "B.Sc. Mathematics",
            "B.Sc. Physics",
            "B.Sc. Chemistry",
            "B.Sc. Botany",
            "B.Sc. Zoology",
            "B.Sc. Computer Science",
            "B.A. History (Tamil Medium)",
            "B.A. History (English Medium)",
            "B.A. Economics",
            "B.Com",
        ],
        "UG_SelfFinancing": [
            "B.A. English Literature",
            "B.Sc. Mathematics",
            "B.Sc. Computer Science",
            "B.Sc. Physical Education",
            "B.A. Tourism Management",
            "B.Com.",
            "B.C.A.",
            "B.B.A.",
        ],
        "PG_Aided": ["M.Sc. Mathematics", "M.Sc. Physics", "M.A. History"],
        "PG_SelfFinancing": [
            "M.A. Tamil",
            "M.A. English",
            "M.Sc. Mathematics",
            "M.Sc. Physics",
            "M.Sc. Chemistry",
            "M.Sc. Botany",
            "M.Sc. Zoology",
            "M.Sc. Computer Science",
            "M.A. Economics",
            "M.Com.",
            "MCA (AICTE Approved)",
            "MBA (AICTE Approved)",
        ],
        "MPhil_Aided": ["History"],
        "MPhil_SelfFinancing": [
            "Tamil",
            "English",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Computer Science",
            "Botany",
            "Zoology",
            "Economics",
            "Commerce",
            "Management Studies",
        ],
        "PhD": [
            "Tamil",
            "English",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Computer Science",
            "Botany",
            "Zoology",
            "History",
            "Economics",
            "Commerce",
            "Management Studies",
        ],
        "Certificate_Diploma": [
            "Air-ticketing and Cargo Management",
            "Business Communication",
            "Computational Biology",
            "Computer Aided Accounting",
            "Entrepreneurship",
            "Export and Import Management",
            "Graphics for Visual Communication",
            "Handicrafts",
            "Herbal Science",
            "Visual Communication",
        ],
        "University_Certificate": [
            "Spoken English (2 Batches)",
            "Spoken Hindi",
            "Driving",
        ],
    }

    def __init__(self, intents_path="intents.json"):
        print("Loading AI Brain...")

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load intents
        with open(intents_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.intents = data["intents"]

        # Store patterns
        self.patterns = []
        self.tags = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.patterns.append(pattern)
                self.tags.append(intent["tag"])

        # Convert patterns to vectors
        self.pattern_embeddings = self.model.encode(self.patterns)

        self.course_catalog = [
            self._normalize_text(course)
            for course_list in self.courses.values()
            for course in course_list
        ]

        print("Chatbot Brain Ready!")

    def _normalize_text(self, text):
        text = text.lower()
        replacements = {
            "b.sc": "bsc",
            "m.sc": "msc",
            "b.a": "ba",
            "m.a": "ma",
            "b.com": "bcom",
            "m.com": "mcom",
            "b.c.a": "bca",
            "m.c.a": "mca",
            "b.b.a": "bba",
            "m.b.a": "mba",
            "m.phil": "mphil",
            "ph.d": "phd",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def course_query_status(self, query):
        course_keywords = {
            "course", "courses", "ba", "bsc", "bcom", "bca", "bba",
            "ma", "msc", "mcom", "mca", "mba", "mphil", "phd",
            "degree", "degrees", "program", "programs",
            "programme", "programmes", "ug", "pg"}
# }

#             "course",
#             "courses",
#             "ba",
#             "bsc",
#             "bcom",
#             "bca",
#             "bba",
#             "ma",
#             "msc",
#             "mcom",
#             "mca",
#             "mba",
#             "mphil",
#             "phd",
#             "degree",
#             "degrees",
#             "program",
#             "programs",
#             "programme",
#             "programmes",
#             "ug",
#             "pg",
#         }


        normalized_query = self._normalize_text(query)
        tokens = normalized_query.split()
        words = set(tokens)
        if not words.intersection(course_keywords):
            return None

        generic_words = {
            "what",
            "which",
            "are",
            "is",
            "the",
            "do",
            "you",
            "have",
            "offer",
            "offered",
            "provide",
            "provided",
            "available",
            "list",
            "show",
            "tell",
            "about",
            "me",
            "all",
            "any",
            "course",
            "courses",
            "program",
            "programs",
            "programme",
            "programmes",
            "ug",
            "pg",
            "ba",
            "bsc",
            "bcom",
            "bca",
            "bba",
            "ma",
            "msc",
            "mcom",
            "mca",
            "mba",
            "mphil",
            "phd",
            "degree",
            "degrees",
        }
        specific_tokens = [token for token in tokens if token not in generic_words]
        if not specific_tokens:
            return None

        for course in self.course_catalog:
            course_words = set(course.split())
            if all(token in course_words for token in specific_tokens):
                return "known"

        return "unknown"

    # ---------- Predict Intent ----------
    def predict_intent(self, user_message):
        user_embedding = self.model.encode([user_message])

        similarities = cosine_similarity(user_embedding, self.pattern_embeddings)[0]

        best_index = np.argmax(similarities)
        confidence = similarities[best_index]
        tag = self.tags[best_index]

        return tag, confidence

    # ---------- Generate Response ----------
    def get_response(self, message):
        tag, confidence = self.predict_intent(message)

        # Sentiment analysis
        sentiment = TextBlob(message).sentiment.polarity
        if message.lower().startswith("my name is"):
            name = message[11:].strip()
            if name:
                return f"Nice to meet you, {name}! How can I assist you today?"
            return "Nice to meet you! How can I assist you today?"

        # If it is a course-name query and not in catalog, return explicit response.
        course_status = self.course_query_status(message)
        if course_status == "unknown":
            return "Sorry, we currently dont provide that course"

        # Confidence threshold + sentiment check
        if confidence < 0.65 or sentiment < -0.2:
            return random.choice(
                [
                    "Sorry, I didn't understand that. Could you rephrase?",
                    "I'm not sure I got that. Try asking about courses, admissions, or fees.",
                    "Apologies - can you ask that differently?",
                ]
            )

        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        return "Something went wrong."


if __name__ == "__main__":
    bot = NMCCChatbot()
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Bot: Goodbye!")
            break
        response = bot.get_response(message)
        print("Bot:", response)
