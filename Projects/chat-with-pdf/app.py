from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import load_rag_chain
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
rag_chain = None


def extract_and_split(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150
    )
    return splitter.split_text(text)


@app.route("/load", methods=["POST"])
def load_pdf():
    global rag_chain

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    
    # Check if file is actually selected
    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Check if file is a PDF
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400

    try:
        # Extract and split the PDF
        chunks = extract_and_split(pdf_file)
        
        if not chunks:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # Debug: Show chunk info
        print(f"\n{'='*50}")
        print(f"PDF UPLOADED: {pdf_file.filename}")
        print(f"TOTAL CHUNKS CREATED: {len(chunks)}")
        print(f"FIRST CHUNK PREVIEW: {chunks[0][:200]}...")
        print(f"{'='*50}\n")
        
        # Load the RAG chain
        rag_chain = load_rag_chain(chunks)
        
        return jsonify({
            "message": "PDF loaded successfully",
            "chunks": len(chunks)
        })
    
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    global rag_chain

    if rag_chain is None:
        return jsonify({"answer": "Please upload a PDF first"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"answer": "No question provided"}), 400
    
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a valid question"}), 400

    try:
        # Debug: Show question
        print(f"\n{'='*50}")
        print(f"USER QUESTION: {question}")
        print(f"{'='*50}")
        
        # Invoke the RAG chain and get the answer
        answer = rag_chain.invoke(question)
        
        # Ensure answer is a string
        answer_text = str(answer).strip() if answer else "I couldn't generate an answer."
        
        # Debug: Show answer
        print(f"\nANSWER GENERATED: {answer_text[:200]}...")
        print(f"{'='*50}\n")
        
        return jsonify({"answer": answer_text})
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
