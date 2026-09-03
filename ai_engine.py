from collections import Counter, defaultdict
from pathlib import Path
import math
import pickle
import re


BASE_DIR = Path(__file__).parent
CATEGORY_MODEL_PATH = BASE_DIR / "models" / "category_model.pkl"
PRIORITY_MODEL_PATH = BASE_DIR / "models" / "priority_model_v2.pkl"
CATEGORY_TRAINING_DATA = [
    ("forgot my password and cannot log in", "Access"),
    ("password reset link is not working", "Access"),
    ("account is locked after login attempts", "Access"),
    ("cannot access company email account", "Access"),
    ("permission denied while opening shared folder", "Access"),

    ("VPN connection failed from home", "Network"),
    ("Wi-Fi is not connecting on my laptop", "Network"),
    ("internet is very slow in the office", "Network"),
    ("cannot connect to company server", "Network"),
    ("network drive is not available", "Network"),

    ("laptop keyboard keys are not working", "Hardware"),
    ("computer screen is flickering", "Hardware"),
    ("mouse stopped working", "Hardware"),
    ("laptop battery drains very quickly", "Hardware"),
    ("printer is not printing documents", "Hardware"),

    ("Microsoft Teams application keeps crashing", "Software"),
    ("unable to install required application", "Software"),
    ("Excel is not opening on my computer", "Software"),
    ("browser crashes while opening websites", "Software"),
    ("software update failed to install", "Software"),

    ("need general technical support", "Other"),
    ("requesting an additional monitor", "Other"),
    ("where can I find IT policy documents", "Other"),
    ("need help with office technology", "Other"),
    ("not sure which support category to choose", "Other"),
]

PRIORITY_TRAINING_DATA = [
    ("all company systems are down", "Critical"),
    ("production server outage affecting every employee", "Critical"),
    ("suspected data breach and unauthorized access", "Critical"),
    ("payment platform is unavailable for all customers", "Critical"),
    ("entire office network is down", "Critical"),
    ("ransomware attack detected on company systems", "Critical"),

    ("VPN not working before an urgent client meeting", "High"),
    ("cannot access email to send an important client report", "High"),
    ("laptop will not start before a presentation", "High"),
    ("account locked and project deadline is today", "High"),
    ("server unavailable for an urgent customer issue", "High"),
    ("cannot work because remote access has failed", "High"),

    ("forgot password and cannot log in", "Medium"),
    ("password reset request without an urgent deadline", "Medium"),
    ("cannot access email but work is not urgent", "Medium"),
    ("printer is not printing documents", "Medium"),
    ("Microsoft Teams application crashes sometimes", "Medium"),
    ("laptop is running slowly", "Medium"),
    ("software installation failed", "Medium"),
    ("Excel is showing an unexpected error", "Medium"),
    ("VPN connection drops occasionally", "Medium"),

    ("requesting an extra monitor", "Low"),
    ("need a new mouse", "Low"),
    ("question about IT policy documents", "Low"),
    ("requesting optional software installation", "Low"),
    ("need help changing desktop wallpaper", "Low"),
    ("requesting a laptop bag", "Low"),
    ("need a replacement keyboard with no urgency", "Low"),
    ("general question about IT support", "Low"),
]


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def train_model(training_data):
    label_document_counts = Counter()
    label_word_counts = defaultdict(Counter)
    label_total_words = Counter()
    vocabulary = set()

    for text, label in training_data:
        label_document_counts[label] += 1

        for word in tokenize(text):
            label_word_counts[label][word] += 1
            label_total_words[label] += 1
            vocabulary.add(word)

    return {
        "labels": list(label_document_counts.keys()),
        "label_document_counts": dict(label_document_counts),
        "label_word_counts": {
            label: dict(word_counts)
            for label, word_counts in label_word_counts.items()
        },
        "label_total_words": dict(label_total_words),
        "vocabulary": list(vocabulary),
        "total_documents": len(training_data),
    }


def save_model(model, model_path):
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_or_train_model(model_path, training_data):
    if model_path.exists():
        with open(model_path, "rb") as file:
            return pickle.load(file)

    model = train_model(training_data)
    save_model(model, model_path)

    return model


def predict(model, ticket_text):
    words = tokenize(ticket_text)
    vocabulary = set(model["vocabulary"])
    vocabulary_size = len(vocabulary)
    labels = model["labels"]
    scores = {}

    for label in labels:
        document_count = model["label_document_counts"][label]

        score = math.log(
            document_count / model["total_documents"]
        )

        word_counts = model["label_word_counts"][label]
        total_words = model["label_total_words"][label]

        for word in words:
            if word in vocabulary:
                probability = (
                    word_counts.get(word, 0) + 1
                ) / (total_words + vocabulary_size)

                score += math.log(probability)

        scores[label] = score

    largest_score = max(scores.values())

    probabilities = {
        label: math.exp(score - largest_score)
        for label, score in scores.items()
    }

    total_probability = sum(probabilities.values())

    probabilities = {
        label: value / total_probability
        for label, value in probabilities.items()
    }

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]

    return prediction, confidence


def predict_category(ticket_text):
    model = load_or_train_model(
        CATEGORY_MODEL_PATH,
        CATEGORY_TRAINING_DATA
    )

    return predict(model, ticket_text)


def predict_priority(ticket_text):
    model = load_or_train_model(
        PRIORITY_MODEL_PATH,
        PRIORITY_TRAINING_DATA
    )

    return predict(model, ticket_text)

def assess_sla_risk(ticket_text, priority):
    text = ticket_text.lower()

    urgent_keywords = [
        "outage",
        "urgent",
        "deadline",
        "unable to work",
        "all employees",
        "data breach",
        "production",
    ]

    if priority == "Critical":
        return "High", "Critical tickets require immediate action."

    if priority == "High" or any(
        keyword in text for keyword in urgent_keywords
    ):
        return "High", "This ticket may miss its SLA without quick action."

    if priority == "Medium":
        return "Medium", "Monitor this ticket and resolve it within the SLA."

    return "Low", "This ticket has a low immediate SLA risk."
KNOWLEDGE_BASE = {
    "Access": [
        "Verify the employee identity according to company policy.",
        "Check whether the account is locked or the password has expired.",
        "Start the approved password-reset or access-request process.",
    ],
    "Network": [
        "Check VPN, Wi-Fi, or Ethernet connectivity.",
        "Ask the user to restart the network connection.",
        "Verify whether other employees have the same issue.",
    ],
    "Hardware": [
        "Check cables, battery level, and physical device connections.",
        "Restart the affected device.",
        "Create a hardware-replacement request if the issue continues.",
    ],
    "Software": [
        "Restart the application and check for available updates.",
        "Verify that the application is installed correctly.",
        "Collect error details before escalating to the application team.",
    ],
    "Other": [
        "Collect more details from the employee.",
        "Identify the correct support category or responsible team.",
        "Create a standard service request if no incident exists.",
    ],
}


def get_suggested_resolution(category, priority):
    steps = list(KNOWLEDGE_BASE.get(
        category,
        KNOWLEDGE_BASE["Other"]
    ))

    if priority in ["High", "Critical"]:
        steps.insert(
            0,
            "Escalate according to the support team's urgent-incident process."
        )

    return steps
def find_similar_resolved_tickets(
    ticket_text,
    ticket_records,
    max_results=3,
):
    input_words = set(tokenize(ticket_text))
    matches = []

    for ticket in ticket_records:
        resolution_note = ticket.get("resolution_note")

        if (
            ticket.get("status") != "Resolved"
            or not resolution_note
            or str(resolution_note).lower() == "nan"
        ):
            continue

        past_ticket_text = (
            f"{ticket.get('issue_title', '')} "
            f"{ticket.get('description', '')}"
        )

        past_words = set(tokenize(past_ticket_text))

        if not input_words or not past_words:
            continue

        similarity_score = len(input_words & past_words) / len(
            input_words | past_words
        )

        if similarity_score > 0:
            matches.append({
                "ticket_id": ticket["ticket_id"],
                "issue_title": ticket["issue_title"],
                "resolution_note": resolution_note,
                "similarity_score": similarity_score,
            })

    return sorted(
        matches,
        key=lambda ticket: ticket["similarity_score"],
        reverse=True,
    )[:max_results]
if __name__ == "__main__":
    test_ticket = "VPN failed and I cannot work from home."

    category, category_confidence = predict_category(test_ticket)
    priority, priority_confidence = predict_priority(test_ticket)

    print(f"Ticket: {test_ticket}")
    print(f"Predicted category: {category} ({category_confidence:.0%})")
    print(f"Predicted priority: {priority} ({priority_confidence:.0%})")