"""
CareerMate - Sample PDF Generator
Generates realistic career documents and test case PDFs for demonstrations and grading.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def create_resume_pdf(filename: str, title: str, sections: list):
    path = SAMPLES_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=50, rightMargin=50, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0B132B")
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#4F46E5"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    story = [
        Paragraph(title, title_style),
        Spacer(1, 10)
    ]

    for heading, text in sections:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Generated: {path.name}")

def main():
    # 1. Alex Smith - Software Engineer Resume
    create_resume_pdf(
        "Alex_Software_Engineer_Resume.pdf",
        "Alex Smith | Senior Software Engineer",
        [
            ("Professional Summary",
             "Results-driven Senior Software Engineer with 5+ years of experience architecting microservices, cloud applications, and high-throughput REST APIs. Proven expertise in Python, Flask, Docker, Kubernetes, and distributed database systems."),
            ("Core Technical Skills",
             "Languages: Python, JavaScript, TypeScript, Go, SQL<br/>"
             "Frameworks: Flask, FastAPI, React, Node.js<br/>"
             "Cloud & DevOps: AWS (EC2, S3, RDS), Docker, Kubernetes, GitHub Actions, Terraform<br/>"
             "Databases: PostgreSQL, Redis, MongoDB"),
            ("Professional Experience",
             "<b>Senior Software Engineer | CloudTech Solutions (2022 - Present)</b><br/>"
             "- Designed and deployed scalable containerized microservices handling 2M+ requests daily.<br/>"
             "- Optimized PostgreSQL queries resulting in a 35% reduction in latency.<br/>"
             "- Mentored 6 junior engineers and spearheaded agile sprint planning.<br/><br/>"
             "<b>Software Developer | NextGen Apps (2019 - 2022)</b><br/>"
             "- Developed interactive frontend interfaces using React and backend REST APIs using Python/Flask.<br/>"
             "- Integrated secure OAuth2 authentication and automated test suites with 92% code coverage."),
            ("Education & Degrees",
             "Bachelor of Technology (B.Tech) in Computer Science & Engineering<br/>"
             "State Technical University | CGPA: 3.8 / 4.0 | Graduated Magna Cum Laude"),
            ("Certifications",
             "AWS Certified Solutions Architect – Associate (2024)<br/>"
             "Certified Kubernetes Application Developer (CKAD) (2023)")
        ]
    )

    # 2. Priya Sharma - Data Scientist & AI Resume
    create_resume_pdf(
        "Priya_Sharma_Data_Scientist_Resume.pdf",
        "Priya Sharma | AI & Machine Learning Scientist",
        [
            ("Professional Summary",
             "Machine Learning Specialist and Data Scientist with deep expertise in generative AI, large language models (LLMs), natural language processing (NLP), and predictive modeling. Passionate about applying AI to solve real-world problems."),
            ("Technical Competencies",
             "Machine Learning: PyTorch, TensorFlow, Scikit-Learn, Hugging Face, XGBoost<br/>"
             "Data Science & Analytics: Pandas, NumPy, SciPy, Matplotlib, SQL, BigQuery<br/>"
             "MLOps & Tools: MLflow, Docker, ChromaDB, Weights & Biases, Git<br/>"
             "Domains: Generative AI, Retrieval-Augmented Generation (RAG), Computer Vision, Time-Series Forecasting"),
            ("Work Experience",
             "<b>Lead AI Engineer | DeepCognition Labs (2023 - Present)</b><br/>"
             "- Engineered an enterprise RAG knowledge engine utilizing ChromaDB and Gemini embeddings.<br/>"
             "- Fine-tuned open-source LLMs on custom industry datasets, increasing response accuracy by 28%.<br/><br/>"
             "<b>Data Scientist | FinTech Analytics (2021 - 2023)</b><br/>"
             "- Built real-time fraud detection algorithms saving over $1.2M annually.<br/>"
             "- Created automated data pipelines processing 10TB+ daily financial transactions."),
            ("Education & Academic Background",
             "Master of Science (M.S.) in Data Science & Artificial Intelligence<br/>"
             "Institute of Technology | Focus: Deep Learning & Statistical Inference")
        ]
    )

    # 3. Off-Topic Document (Cooking Recipes - for testing rejection)
    create_resume_pdf(
        "Grandmas_Kitchen_Recipes_OffTopic.pdf",
        "Grandma's Best Dessert Recipes",
        [
            ("Chocolate Fudge Cake",
             "Ingredients: 2 cups all-purpose flour, 2 cups sugar, 3/4 cup unsweetened cocoa powder, 2 teaspoons baking powder, 1 1/2 teaspoons baking soda, 1 teaspoon salt, 1 cup milk, 1/2 cup vegetable oil, 2 large eggs, 2 teaspoons vanilla extract, 1 cup boiling water.<br/><br/>"
             "Instructions: Preheat oven to 350 degrees F. Grease and flour two 9-inch round baking pans. In a large bowl, stir together dry ingredients. Add eggs, milk, oil and vanilla; beat on medium speed for 2 minutes. Stir in boiling water. Bake 30 to 35 minutes."),
            ("Classic Italian Lasagna",
             "Layer lasagna noodles with ricotta cheese, tomato sauce, cooked ground beef, mozzarella, and parmesan cheese. Bake at 375 degrees F for 45 minutes until cheese is bubbly and golden brown.")
        ]
    )

    # 4. Scanned / Empty Document (for testing scanned detection)
    blank_path = SAMPLES_DIR / "Scanned_Image_Sample.pdf"
    doc = SimpleDocTemplate(str(blank_path), pagesize=letter)
    # A single dot so it's less than 15 words
    doc.build([Paragraph(".", getSampleStyleSheet()["Normal"])])
    print(f"Generated: {blank_path.name}")

if __name__ == "__main__":
    main()
