
import streamlit as st
import os
import json

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
from pydantic import BaseModel


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI-Powered Question Paper Generator")
st.write("Upload your study material and generate MCQs using Gemini AI.")


# --------------------------------------------------
# CHECK API KEY
# --------------------------------------------------

if api_key:
    st.success("Gemini API key loaded successfully! ✅")
else:
    st.error(
        "Gemini API key not found ❌\n\n"
        "Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()


# --------------------------------------------------
# CREATE GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# MCQ DATA MODEL
# --------------------------------------------------

class MCQ(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: str


class QuestionPaper(BaseModel):
    questions: list[MCQ]


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📚 Upload your study material",
    type=["pdf"]
)


if uploaded_file:

    st.success(f"File uploaded: {uploaded_file.name} ✅")


    # --------------------------------------------------
    # EXTRACT TEXT FROM PDF
    # --------------------------------------------------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"


    # Remove unnecessary spaces
    text = text.strip()


    # Check whether text was extracted
    if not text:

        st.error(
            "Could not extract text from this PDF. "
            "Please upload a text-based PDF."
        )

        st.stop()


    # --------------------------------------------------
    # SHOW EXTRACTED CONTENT
    # --------------------------------------------------

    st.subheader("📖 Extracted Study Material")

    with st.expander("View extracted content"):
        st.text_area(
            "PDF Content",
            text,
            height=300
        )


    # --------------------------------------------------
    # QUESTION PAPER SETTINGS
    # --------------------------------------------------

    st.subheader("⚙️ Question Paper Settings")


    num_questions = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )


    difficulty = st.selectbox(
        "Difficulty Level",
        ["Easy", "Medium", "Hard"]
    )


    question_type = st.selectbox(
        "Question Type",
        ["MCQ"]
    )


    marks = st.number_input(
        "Marks per Question",
        min_value=1,
        max_value=20,
        value=1,
        step=1
    )


    # --------------------------------------------------
    # SHOW SETTINGS
    # --------------------------------------------------

    st.write("### Selected Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Questions:** {num_questions}")
        st.write(f"**Difficulty:** {difficulty}")

    with col2:
        st.write(f"**Type:** {question_type}")
        st.write(f"**Marks:** {marks}")


    # --------------------------------------------------
    # GENERATE BUTTON
    # --------------------------------------------------

    if st.button(
        "🚀 Generate Question Paper",
        type="primary"
    ):

        # Limit extremely large PDFs
        # to keep the prompt manageable.
        material = text[:50000]


        # --------------------------------------------------
        # CREATE AI PROMPT
        # --------------------------------------------------

        prompt = f"""
You are an expert university examination question-paper generator.

Your task is to generate multiple-choice questions ONLY from
the study material provided below.

IMPORTANT RULES:

1. Use only information found in the study material.
2. Do not invent facts that are not present in the material.
3. Generate exactly {num_questions} questions.
4. Difficulty level: {difficulty}.
5. Question type: MCQ.
6. Each question must have exactly four options.
7. Only one option should be correct.
8. Make the incorrect options plausible.
9. Do not repeat questions.
10. Questions should cover different parts of the study material
    whenever possible.
11. Provide a short explanation for the correct answer.
12. The correct_answer must contain the complete text of the
    correct option.
13. Return the result using the requested JSON structure.

Each question has {marks} mark(s).

STUDY MATERIAL:
----------------
{material}
----------------
"""


        # --------------------------------------------------
        # CALL GEMINI
        # --------------------------------------------------

        try:

            with st.spinner(
                "🤖 Gemini is reading the material and generating questions..."
            ):

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QuestionPaper,
                        temperature=0.4,
                    ),
                )


            # --------------------------------------------------
            # PARSE GEMINI RESPONSE
            # --------------------------------------------------

            question_paper = QuestionPaper.model_validate_json(
                response.text
            )


            # Save generated questions in session state
            st.session_state["question_paper"] = question_paper


        except Exception as e:

            st.error(
                "❌ Error while generating questions."
            )

            st.exception(e)


    # --------------------------------------------------
    # DISPLAY GENERATED QUESTIONS
    # --------------------------------------------------

    if "question_paper" in st.session_state:

        question_paper = st.session_state["question_paper"]


        st.divider()

        st.header("📝 Generated Question Paper")


        # Paper header
        st.markdown(
            f"""
### Multiple Choice Questions

**Total Questions:** {len(question_paper.questions)}

**Marks per Question:** {marks}

**Total Marks:** {len(question_paper.questions) * marks}

**Difficulty:** {difficulty}

---
"""
        )


        # --------------------------------------------------
        # DISPLAY EACH MCQ
        # --------------------------------------------------

        for index, mcq in enumerate(
            question_paper.questions,
            start=1
        ):

            st.markdown(
                f"### Q{index}. {mcq.question}"
            )

            st.write(
                f"**A.** {mcq.option_a}"
            )

            st.write(
                f"**B.** {mcq.option_b}"
            )

            st.write(
                f"**C.** {mcq.option_c}"
            )

            st.write(
                f"**D.** {mcq.option_d}"
            )


            # Show answer in an expandable section
            with st.expander("✅ Show Answer"):

                st.success(
                    f"Correct Answer: {mcq.correct_answer}"
                )

                st.info(
                    f"Explanation: {mcq.explanation}"
                )


            st.divider()


        # --------------------------------------------------
        # DOWNLOAD JSON
        # --------------------------------------------------

        json_data = json.dumps(
            question_paper.model_dump(),
            indent=4
        )


        st.download_button(
            label="⬇️ Download Question Paper JSON",
            data=json_data,
            file_name="generated_question_paper.json",
            mime="application/json"
        )
