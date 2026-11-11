"""
CareerMate - Domain Validator
Enforces career-only topic restriction across all incoming questions and documents.
"""
import re
from typing import Tuple
from backend.config import Config

class DomainValidator:
    """
    Validates whether a query falls strictly within the career, education,
    and professional development domain.
    """

    # Direct greetings and capability inquiries (allowed to introduce CareerMate)
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
        r"^who are you\??$",
        r"^what can you do\??$",
        r"^help\??$",
        r"^what is careermate\??$",
        r"^how can you help me\??$"
    ]

    # Explicit Jailbreak attempts
    JAILBREAK_PATTERNS = [
        r"\bignore (all )?(previous )?(instructions|rules|prompts)\b",
        r"\bforget (all )?(your )?(rules|instructions)\b",
        r"\bbypass (rules|filters|guidelines)\b",
        r"\bjailbreak\b",
        r"\bact as (a )?(pirate|movie star|politician|chef|gamer)\b"
    ]

    # Strong Career/Education/IT Domain Keywords
    CAREER_KEYWORDS = {
        # Core Career terms
        "career", "careers", "job", "jobs", "profession", "occupation", "employment",
        "intern", "internship", "internships", "workplace", "resume", "resumes", "cv",
        "curriculum vitae", "cover letter", "portfolio", "interview", "interviews",
        "hiring", "recruiter", "linkedin", "networking", "salary", "salaries",
        "compensation", "promotion", "promotions", "appraisal", "offer letter",
        "negotiation", "job search", "switch career", "career switch", "career transition",
        "career path", "career roadmap", "fresher", "experienced", "experience",

        # Education & Learning
        "education", "degree", "degrees", "college", "university", "bachelor", "master",
        "phd", "btech", "mtech", "bsc", "msc", "bca", "mca", "bootcamp", "course", "courses",
        "curriculum", "syllabus", "certification", "certifications", "certificate",
        "learning", "study", "scholarship", "gate", "gre", "toefl", "ielts", "campus",
        "placement", "placements", "gpa", "cgpa", "academics",

        # Skills & Growth
        "skill", "skills", "soft skills", "hard skills", "upskill", "reskill", "leadership",
        "problem solving", "communication", "time management", "aptitude", "technical skills",

        # Tech, AI, ML, Data Science & Software
        "software", "programming", "coding", "developer", "developers", "engineer", "engineers",
        "frontend", "backend", "fullstack", "full stack", "data science", "data scientist",
        "data analyst", "analytics", "machine learning", "deep learning", "ai",
        "artificial intelligence", "nlp", "computer vision", "llm", "generative ai",
        "python", "java", "javascript", "react", "node", "sql", "database",
        "cloud", "aws", "azure", "gcp", "devops", "docker", "kubernetes",
        "cybersecurity", "security", "git", "github", "system design", "dsa",
        "data structures", "algorithms", "leetcode", "tech stack", "it career", "it careers"
    }

    # Explicitly forbidden out-of-domain topics
    OFF_TOPIC_PATTERNS = [
        # Entertainment, Movies, Celebrity gossip, Music
        r"\b(movie|movies|cinema|actor|actors|actress|bollywood|hollywood|netflix|box office|song|songs|singer|music band|album)\b",
        # Sports & Athletics
        r"\b(cricket|football|soccer|fifa|world cup|ipl|premier league|nba|tennis|badminton|olympics|score|match|game score|batting|bowling)\b",
        # Politics, Elections, Non-tech Government
        r"\b(election|elections|political|politician|prime minister|president|parliament|congress|bjp|democrat|republican|voting|vote)\b",
        # Food & Cooking
        r"\b(recipe|recipes|cook|cooking|ingredients|pizza|burger|biryani|cake|bake|baking)\b",
        # Astrology, Dating & Romance
        r"\b(horoscope|zodiac|astrology|dating|tinder|boyfriend|girlfriend|love life|breakup|crush)\b",
        # Weather & News
        r"\b(weather|forecast|rain|rainy|temperature|humidity)\b",
        # Video games walkthrough / cheats
        r"\b(gta|pubg|fortnite|minecraft|game cheat|playstation|xbox)\b"
    ]

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Checks if the query is a friendly greeting or capability inquiry."""
        clean = query.strip().lower()
        return any(re.search(pat, clean) for pat in cls.GREETING_PATTERNS)

    @classmethod
    def is_jailbreak(cls, query: str) -> bool:
        """Detects prompt injection or jailbreak attempts."""
        clean = query.strip().lower()
        return any(re.search(pat, clean) for pat in cls.JAILBREAK_PATTERNS)

    @classmethod
    def is_explicitly_off_topic(cls, query: str) -> bool:
        """
        Quick check for clearly unrelated queries (sports, movies, politics, recipes).
        Safeguards false positives if combined with career terms (e.g., 'sports journalism career').
        """
        clean = query.strip().lower()

        # If it explicitly includes career/job keywords, let it pass to AI context evaluation
        if any(term in clean for term in ["career", "job", "degree", "study", "skills", "interview", "resume", "become a"]):
            return False

        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, clean):
                return True
        return False

    @classmethod
    def contains_career_keywords(cls, query: str) -> bool:
        """Checks if the query contains any recognized career/education/tech domain keywords."""
        tokens = set(re.findall(r"\b[a-z0-9+#.-]+\b", query.lower()))
        return bool(tokens & cls.CAREER_KEYWORDS)

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, str]:
        """
        Validates the user query.
        Returns:
            Tuple (is_valid: bool, reason_or_reply: str)
            If is_valid is True, query is accepted.
            If is_valid is False, reason_or_reply contains the required polite rejection.
        """
        if not query or not query.strip():
            return False, "Please enter a question so I can assist you with your career goals."

        clean = query.strip()

        # Check jailbreak attempts
        if cls.is_jailbreak(clean):
            return False, Config.DOMAIN_REJECTION_MESSAGE

        # Handle greetings nicely
        if cls.is_greeting(clean):
            return True, "GREETING"

        # Check explicit off-topic patterns
        if cls.is_explicitly_off_topic(clean):
            return False, Config.DOMAIN_REJECTION_MESSAGE

        # If it contains career keywords, approve
        if cls.contains_career_keywords(clean):
            return True, "APPROVED"

        # Queries that might be general chat with no career intent
        words = clean.split()
        if len(words) <= 3 and not cls.contains_career_keywords(clean):
            return False, Config.DOMAIN_REJECTION_MESSAGE

        # If longer query has zero career keywords and matches off-topic terms
        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, clean.lower()):
                return False, Config.DOMAIN_REJECTION_MESSAGE

        return True, "APPROVED"
