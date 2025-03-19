import streamlit as st
import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words("english"))

DEADLINE_PATTERNS = [
    # Specific Days of the Week
    r"\b(by|before|on|until|till|at|after|during)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",

    # Time-Specific Deadlines
    r"\b(by|before|on|until|till|at|after|during)\s+\d{1,2}\s*(AM|PM|am|pm|noon|midnight|morning|afternoon|evening|night)\b",

    # Relative Deadlines
    r"\b(by|before|on|until|till|at|after|during|within)\s+(next|this|upcoming|following|last|previous)\s*(week|month|year|evening|morning|noon|night)\b",
    r"\bwithin\s+the\s+next\s+\d+\s*(hours|days|weeks|months|years)\b",
    r"\b(by|before|on|until|till|at|after|during)\s+tomorrow\b",

    # Exact Date Expressions
    r"\b(by|before|on|until|till|at|after|during)\s+\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\b(by|before|on|until|till|at|after|during)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?\b",

    # Numeric Date Formats (MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD)
    r"\b(by|before|on|until|till|at|after|during)\s+\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b(by|before|on|until|till|at|after|during)\s+\d{1,2}-\d{1,2}-\d{2,4}\b",
    r"\b(by|before|on|until|till|at|after|during)\s+\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD format

    # General Deadlines
    r"\b(by|before|on|until|till|at|after|during)\s+end\s+of\s+(day|week|month|quarter|year)\b",
    
    # EOD, COB, ASAP, and Urgent Terms
    r"\b(by|before|on|until|till|at|after|during)\s+EOD\b",  # End of Day
    r"\b(by|before|on|until|till|at|after|during)\s+COB\b",  # Close of Business
    r"\bASAP\b",  # As soon as possible
    r"\bimmediately\b",
    r"\burgently\b",

    # Task Completion Deadlines
    r"\bno\s+later\s+than\b",  # Example: "Submit no later than Friday"
    r"\bdue\s+(on|by)\b",  # Example: "Report is due on Wednesday"
    r"\bsubmit\s+by\b",
    r"\bcomplete\s+before\b",
    r"\bfinish\s+by\b",
    r"\bturn\s+in\s+before\b",
    r"\bexpected\s+by\b",
    r"\brequired\s+by\b",
    r"\bdeliver\s+by\b",

    # Estimated Time of Arrival
    r"\bETA\b",

    # Deadlines within a time frame
    r"\bwithin\s+\d+\s*(hours|days|weeks|months|years)\b",
    r"\b(to|before|after|until|till)\s+\d+\s*(hours|days|weeks|months|years)\b"
]



CATEGORIES = {
    "Work": [
        "report", "presentation", "budget", "proposal", "documentation", "email", "submission",
        "task", "project", "plan", "strategy", "agenda", "workflow", "deadline", "deliverables",
        "assignment", "workload", "summary", "briefing", "scheduling", "timeline"
    ],
    "Technical": [
        "fix", "update", "install", "configure", "debug", "server", "software", "deployment",
        "patch", "code review", "troubleshoot", "diagnose", "hardware", "firmware", "network",
        "API", "integration", "script", "data pipeline", "database", "backend", "frontend"
    ],
    "Software Development": [
        "coding", "programming", "development", "testing", "debugging", "refactoring",
        "release", "commit", "push", "pull request", "version control", "GitHub", "JIRA",
        "unit test", "automation", "CI/CD", "containerization", "Docker", "Kubernetes"
    ],
    "AI & Data Science": [
        "machine learning", "deep learning", "NLP", "computer vision", "data cleaning",
        "data preprocessing", "model training", "fine-tuning", "inference", "AI model",
        "data visualization", "big data", "SQL", "data warehouse", "ETL pipeline", "data scraping",
        "web scraping", "API integration", "model deployment", "hyperparameter tuning"
    ],
    "Meeting": [
        "schedule", "call", "meeting", "discussion", "appointment", "conference",
        "stand-up", "scrum", "brainstorming", "review", "1-on-1", "client call", "sync-up",
        "team meeting", "weekly update", "presentation", "video call", "Zoom", "Teams",
        "negotiation", "board meeting", "town hall"
    ],
    "Deadline-Oriented": [
        "submit", "complete", "finalize", "review", "approve", "send",
        "due date", "deadline", "urgent", "high priority", "follow-up", "wrap up",
        "last-minute", "time-sensitive", "deliverable", "milestone", "submission",
        "turn in", "expedite", "end of day", "ASAP"
    ],
    "Personal": [
        "exercise", "meditate", "cook", "shop", "clean", "grocery", "laundry",
        "self-care", "reading", "journaling", "hobby", "meal prep", "fitness",
        "walking", "running", "sleep schedule", "meditation", "hydration", "well-being"
    ],
    "Administrative": [
        "paperwork", "forms", "filing", "process", "policy", "regulation",
        "document signing", "approval", "compliance", "audit", "record keeping",
        "scheduling", "clerical", "data entry", "clerical work", "email response",
        "documentation", "standard operating procedures", "admin tasks"
    ],
    "Learning & Development": [
        "study", "read", "course", "training", "workshop", "webinar",
        "certification", "skill-building", "e-learning", "self-paced learning",
        "bootcamp", "MOOC", "mentorship", "knowledge sharing", "presentation skills",
        "online class", "skill enhancement", "research", "case study", "tutorial"
    ],
    "Finance": [
        "invoice", "payment", "budget", "expense", "reimbursement",
        "salary", "payroll", "tax", "banking", "investment", "loan", "mortgage",
        "financial report", "accounting", "profit & loss", "billing", "accounts payable",
        "accounts receivable", "bookkeeping", "financial statement", "audit"
    ],
    "Sales & Marketing": [
        "campaign", "advertisement", "branding", "promotion", "outreach",
        "lead generation", "client acquisition", "sales pitch", "sales funnel",
        "SEO", "PPC", "social media marketing", "email marketing", "Google Ads",
        "market research", "customer engagement", "influencer marketing", "content strategy",
        "conversion optimization", "business development"
    ],
    "HR & Recruitment": [
        "interview", "hire", "onboarding", "training", "job description",
        "resume screening", "offer letter", "employee engagement", "exit interview",
        "performance review", "team building", "benefits management", "payroll processing",
        "policy creation", "talent acquisition", "headhunting", "background check",
        "employee relations", "succession planning"
    ],
    "Client Management": [
        "follow-up", "client call", "customer support", "contract",
        "proposal", "client meeting", "negotiation", "partnership", "service request",
        "account management", "CRM", "customer feedback", "escalation", "user support",
        "onboarding", "support ticket", "service-level agreement", "issue resolution"
    ],
    "Health & Wellness": [
        "doctor appointment", "dentist", "therapy", "medication", "checkup",
        "mental health", "yoga", "diet planning", "vitamins", "wellness program",
        "physical therapy", "annual checkup", "eye exam", "nutrition", "rehabilitation",
        "vaccination", "emergency medical care", "health insurance", "exercise routine"
    ],
    "Household & Chores": [
        "grocery shopping", "cleaning", "home repair", "garden", "maintenance",
        "cooking", "laundry", "decluttering", "pet care", "home improvement",
        "pest control", "plumbing", "electrical work", "furniture assembly",
        "renovation", "moving", "storage organization", "landscaping", "housekeeping"
    ],
    "Travel & Logistics": [
        "book flight", "hotel reservation", "visa application", "travel itinerary",
        "transportation", "car rental", "packing", "trip planning", "travel insurance",
        "road trip", "airport transfer", "check-in", "luggage", "boarding pass",
        "map navigation", "business trip", "vacation planning", "passport renewal"
    ],
    "Legal & Compliance": [
        "contract review", "legal paperwork", "audit", "compliance check",
        "intellectual property", "lawsuit", "terms & conditions", "NDAs",
        "risk assessment", "GDPR", "HIPAA compliance", "policy enforcement",
        "due diligence", "corporate law", "liability", "tax law", "fraud prevention"
    ],
    "Product Management": [
        "roadmap", "product strategy", "user research", "feature request",
        "stakeholder alignment", "A/B testing", "wireframing", "prototyping",
        "user experience", "customer feedback", "agile methodology", "MVP development",
        "backlog prioritization", "product launch", "market analysis"
    ],
    "Content Creation": [
        "blog post", "copywriting", "scriptwriting", "video editing", "graphic design",
        "content calendar", "social media post", "SEO optimization", "infographic",
        "podcast recording", "press release", "creative writing", "e-book",
        "newsletter", "thumbnail design", "content marketing"
    ]
}


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def extract_tasks(paragraph):
    sentences = sent_tokenize(paragraph)
    tasks = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        
        action_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        
        who = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
        
        deadline = next((re.search(pattern, sentence, re.IGNORECASE).group(0) for pattern in DEADLINE_PATTERNS if re.search(pattern, sentence, re.IGNORECASE)), "No deadline")
        
        if action_verbs:
            tasks.append({"task": sentence.strip(), "who": who, "deadline": deadline})
    
    return tasks

def train_word2vec(sentences):
    tokenized_sentences = [word_tokenize(preprocess_text(sent)) for sent in sentences]
    if not tokenized_sentences or all(len(sentence) == 0 for sentence in tokenized_sentences):
        return None
    return Word2Vec(tokenized_sentences, vector_size=50, min_count=1, window=5)






def categorize_task(task_description, word2vec_model):
    if not word2vec_model:
        return "General Task"
    words = [word for word in word_tokenize(preprocess_text(task_description)) if word in word2vec_model.wv]
    if not words:
        return "General Task"
    task_vector = np.mean([word2vec_model.wv[word] for word in words], axis=0)
    best_category, best_similarity = "General Task", float("inf")
    for category, keywords in CATEGORIES.items():
        keyword_vectors = [word2vec_model.wv[word] for word in keywords if word in word2vec_model.wv]
        if keyword_vectors:
            category_vector = np.mean(keyword_vectors, axis=0)
            similarity = cosine(task_vector, category_vector)
            if similarity < best_similarity:
                best_similarity, best_category = similarity, category
    return best_category

def categorize_tasks(tasks):
    sentences = [task["task"] for task in tasks]
    word2vec_model = train_word2vec(sentences)
    for task in tasks:
        task["category"] = categorize_task(task["task"], word2vec_model)
    return tasks

# # Main execution
# if __name__ == "__main__":
#     while True:
#         paragraph = input("Enter a paragraph: ") 
#         # Extract tasks
#         tasks = extract_tasks(paragraph)  
#         # Categorize tasks
#         tasks = categorize_tasks(tasks)
#         found = False
#         for task in tasks:
#             if task['deadline'] != "No deadline":
#                 print("\nExtracted Tasks:\n")
#                 print(f"Task: {task['task']}")
#                 print(f"Who: {task['who']}")
#                 print(f"Deadline: {task['deadline']}")
#                 print(f"Category: {task['category']}\n")
#                 found = True
#         if not found:
#             print("\nNo tasks found in the paragraph.\n")
            


if __name__ == "__main__":
        
    st.markdown("<p style='color:orange; font-size:20px;'>Restrictions: No LLMs Used</p>", unsafe_allow_html=True)
    st.title("📝 Task Extractor")
    st.markdown("### Extract deadlines and categorize tasks effortlessly!")

    paragraph = st.text_area("✍️ Enter your text:", height=200)
    if st.button("🔍 Extract Tasks"):
        st.markdown("<p style='color:red; font-size:20px;'>Emergency Deadline Oriented Task's</p>", unsafe_allow_html=True)
        if paragraph.strip():
            tasks = extract_tasks(paragraph)
            tasks = categorize_tasks(tasks)
            if tasks:
                st.subheader("Extracted Tasks:")
                for task in tasks:
                    if task['deadline'] != "No deadline":
                        st.write(f"**Task:** {task['task']}")
                        st.write(f"**Who:** {task['who']}")
                        st.write(f"**Deadline:** {task['deadline']}")
                        st.write(f"**Category:** {task['category']}")
                        st.write("---")
            else:
                st.write("No tasks found in the paragraph.")
        else:
            st.warning("Please enter a paragraph to extract tasks.")
