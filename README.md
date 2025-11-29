# Advanced RAG System Comparison: LangChain vs Custom Implementation

A comprehensive comparison of two Retrieval-Augmented Generation (RAG) systems for the fictional company **Insurellm**, demonstrating the performance improvements achieved through advanced RAG techniques.

## 📊 Project Overview

This project implements and evaluates two different RAG systems:

1. **Basic Implementation** (`implementation/`) - Traditional LangChain-based RAG
2. **Advanced Implementation** (`pro_implementation/`) - Custom RAG with sophisticated techniques

Both systems answer questions about Insurellm using a knowledge base of 76 markdown documents, but the advanced implementation achieves significantly better retrieval and answer quality through intelligent chunking, query optimization, and reranking.
### langChain Retrival
![tapimage](https://github.com/AhmedMostafa3m/RAG_Langchain_vs_advanced_technics/blob/85b453e726cd089dd5577df8ccd5d601c0d189dc/Images/retrival_Eval.png)

### langChain Answer
![tapimage](https://github.com/AhmedMostafa3m/RAG_Langchain_vs_advanced_technics/blob/3cd99a3c2277d612853e5e45c1d2df0d174e5ca5/Images/Answer_Eval.png)

### Custom Retrival
![tapimage](https://github.com/AhmedMostafa3m/RAG_Langchain_vs_advanced_technics/blob/823323cc7f7996df557c624ac5dc9a9508eba772/Images/Advanced_retrieval.png)

### Custom Answer
![tapimage](https://github.com/AhmedMostafa3m/RAG_Langchain_vs_advanced_technics/blob/823323cc7f7996df557c624ac5dc9a9508eba772/Images/Advanced_Answer.png)

## 🎯 Key Improvements

The advanced RAG system introduces several enhancements over the basic LangChain approach:

### 1. **LLM-Powered Intelligent Chunking**
- **Basic**: Fixed-size chunks (500 chars) with 200-char overlap using `RecursiveCharacterTextSplitter`
- **Advanced**: Dynamic, context-aware chunking using GPT-4o-mini
  - Each chunk includes a headline and summary for better retrieval
  - Intelligent overlap based on semantic boundaries
  - Preserves document context and relationships

### 2. **Query Rewriting**
- **Basic**: Uses raw user query directly
- **Advanced**: Rewrites queries to be more specific and retrieval-friendly
  - Considers conversation history
  - Optimizes for knowledge base search
  - Improves semantic matching

### 3. **Multi-Query Retrieval**
- **Basic**: Single query retrieval (k=10)
- **Advanced**: Dual-query approach
  - Retrieves with both original and rewritten queries (k=20 each)
  - Merges results to maximize coverage
  - Reduces missed relevant documents

### 4. **LLM-Based Reranking**
- **Basic**: Uses vector similarity scores only
- **Advanced**: GPT-4o-mini reranks retrieved chunks
  - Evaluates true relevance to the question
  - Returns top 10 most relevant chunks
  - Significantly improves precision

## 🏗️ Architecture

### Basic Implementation (LangChain)

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ RecursiveCharacterText  │
│     Splitter            │
│  (500 chars, 200 overlap)│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  OpenAI Embeddings      │
│  (text-embedding-3-large)│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   ChromaDB Vector Store │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Retrieve Top-K (10)    │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   GPT-4o-mini Answer    │
└─────────────────────────┘
```

### Advanced Implementation (Custom)

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  LLM-Powered Chunking   │
│  (GPT-4o-mini)          │
│  - Headlines            │
│  - Summaries            │
│  - Smart overlap        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  OpenAI Embeddings      │
│  (text-embedding-3-large)│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   ChromaDB Vector Store │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Query Rewriting       │
│   (GPT-4o-mini)         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Dual Query Retrieval   │
│  - Original (k=20)      │
│  - Rewritten (k=20)     │
│  - Merge unique         │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  LLM Reranking          │
│  (GPT-4o-mini)          │
│  - Top 10 chunks        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   GPT-4o-mini Answer    │
└─────────────────────────┘
```

## 📁 Project Structure

```
week5/
├── implementation/              # Basic LangChain RAG
│   ├── ingest.py               # Document ingestion with RecursiveCharacterTextSplitter
│   └── answer.py               # Simple retrieval and answering
│
├── pro_implementation/          # Advanced Custom RAG
│   ├── ingest.py               # LLM-powered intelligent chunking
│   └── answer.py               # Query rewriting, dual retrieval, and reranking
│
├── evaluation/                  # Evaluation framework
│   ├── eval.py                 # Retrieval and answer evaluation metrics
│   └── test.py                 # Test question loader
│
├── evaluator.py                # Gradio UI for running evaluations
├── knowledge-base/             # 76 markdown documents about Insurellm
├── vector_db/                  # Basic implementation vector store
└── preprocessed_db/            # Advanced implementation vector store
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) Groq API key for faster LLM calls

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/advanced-rag-comparison.git
   cd advanced-rag-comparison
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or with uv
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the `llm_engineering` directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here  # Optional
   ```

### Usage

#### 1. Ingest Documents

**Basic Implementation:**
```bash
cd week5/implementation
python ingest.py
```

**Advanced Implementation:**
```bash
cd week5/pro_implementation
python ingest.py
```

#### 2. Ask Questions

**Basic Implementation:**
```python
from implementation.answer import answer_question

answer, docs = answer_question("What is Insurellm's mission?")
print(answer)
```

**Advanced Implementation:**
```python
from pro_implementation.answer import answer_question

answer, docs = answer_question("What is Insurellm's mission?")
print(answer)
```

#### 3. Run Evaluation

**Gradio Dashboard:**
```bash
cd week5
python evaluator.py
```

**CLI Evaluation:**
```bash
cd week5/evaluation
python eval.py 0  # Evaluate test question #0
```

## 📈 Evaluation Metrics

The evaluation system measures both **retrieval quality** and **answer quality**:

### Retrieval Metrics

1. **Mean Reciprocal Rank (MRR)**
   - Measures how quickly relevant documents are found
   - Range: 0-1 (higher is better)
   - Formula: Average of 1/rank for each keyword's first appearance

2. **Normalized Discounted Cumulative Gain (nDCG)**
   - Measures ranking quality with position-based discounting
   - Range: 0-1 (higher is better)
   - Rewards relevant documents appearing earlier

3. **Keyword Coverage**
   - Percentage of expected keywords found in top-k results
   - Range: 0-100% (higher is better)

### Answer Quality Metrics (LLM-as-Judge)

1. **Accuracy** (1-5 scale)
   - How factually correct is the answer?
   - Compared against reference answer

2. **Completeness** (1-5 scale)
   - Does it address all aspects of the question?
   - Covers all information from reference answer

3. **Relevance** (1-5 scale)
   - How well does it answer the specific question?
   - No extraneous information

## 🎓 Key Learnings

### When to Use Advanced RAG

The advanced implementation is beneficial when:
- **Document complexity is high** - Technical docs, legal documents, research papers
- **Precision is critical** - Medical, legal, or financial applications
- **Query ambiguity is common** - Users ask vague or multi-part questions
- **Context matters** - Answers require understanding document relationships

### Trade-offs

| Aspect | Basic RAG | Advanced RAG |
|--------|-----------|--------------|
| **Setup Complexity** | Low | High |
| **Ingestion Time** | Fast (~30s) | Slow (~10-20 min) |
| **Query Latency** | Low (~1-2s) | Higher (~3-5s) |
| **Cost per Query** | Low | Higher (3-4x LLM calls) |
| **Retrieval Quality** | Good | Excellent |
| **Answer Quality** | Good | Excellent |

### Best Practices

1. **Start Simple** - Begin with basic RAG, measure performance
2. **Identify Pain Points** - Use evaluation metrics to find weaknesses
3. **Iterate Incrementally** - Add one advanced technique at a time
4. **Monitor Costs** - LLM-powered chunking and reranking increase API costs
5. **Cache When Possible** - Reuse rewritten queries and reranked results
6. **Tune Hyperparameters** - Adjust k values, chunk sizes, and overlap

## 🛠️ Technologies Used

- **LangChain** - RAG framework and utilities
- **ChromaDB** - Vector database
- **OpenAI** - Embeddings (text-embedding-3-large) and LLM (GPT-4o-mini)
- **LiteLLM** - Unified LLM API interface
- **Pydantic** - Data validation and structured outputs
- **Gradio** - Evaluation dashboard UI
- **Tenacity** - Retry logic for API calls

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/yourprofile).

## 🙏 Acknowledgments

- Based on the LLM Engineering course by Ed Donner
- Inspired by advanced RAG techniques from the AI research community
- Built for the fictional company Insurellm as a learning project

---

**Note**: This is an educational project demonstrating RAG system design and evaluation. The company "Insurellm" and its knowledge base are fictional.




