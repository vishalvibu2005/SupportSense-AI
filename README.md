# SupportSense AI

SupportSense AI is an intelligent IT service-desk platform that helps support teams create, manage, analyze, and resolve IT support tickets.

## Features

- Create and manage IT support tickets
- Search and filter tickets by category, priority, and status
- Predict ticket category using a Naive Bayes text-classification model
- Predict ticket priority: Low, Medium, High, or Critical
- Assess SLA risk using transparent business rules
- Provide knowledge-base troubleshooting recommendations
- Find similar previously resolved tickets and their resolution notes
- Export filtered ticket reports as CSV files
- Track ticket status: Open, In Progress, and Resolved

## AI Workflow

```text
Ticket Description
        ↓
Category Prediction
        ↓
Priority Prediction
        ↓
SLA Risk Assessment
        ↓
Knowledge-Base Recommendation
        ↓
Similar Past Resolved Tickets
```

## Technology Stack

- Python
- Streamlit
- SQLite
- pandas
- Naive Bayes text classification
- Git and GitHub

## Project Structure

```text
SupportSense-AI/
├── app.py              # Streamlit web application
├── ai_engine.py        # AI prediction and recommendation logic
├── requirements.txt    # Required Python packages
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

## How to Run Locally

1. Clone or download the repository.

2. Create and activate a virtual environment.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
python -m streamlit run app.py
```

5. Open the local URL shown in the terminal.

## Future Improvements

- Train models using a larger real-world IT ticket dataset
- Add user authentication and role-based access
- Deploy the application to the cloud
- Add semantic search for more accurate similar-ticket matching
- Build advanced SLA and support-team analytics

## Author

K Vishal Vibu  
B.E. Artificial Intelligence and Machine Learning