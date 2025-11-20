<div align="center">
<pre>
███╗   ███╗ ██████╗ ██╗   ██╗██╗███████╗    ██████╗  █████╗  ██████╗ 
████╗ ████║██╔═══██╗██║   ██║██║██╔════╝    ██╔══██╗██╔══██╗██╔════╝ 
██╔████╔██║██║   ██║██║   ██║██║█████╗      ██████╔╝███████║██║  ███╗
██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██║██╔══╝      ██╔══██╗██╔══██║██║   ██║
██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║███████╗    ██║  ██║██║  ██║╚██████╔╝
╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
-----------------------------------------------------------------------
Project from a course i took on Boot.dev where i build a RAG from the ground up :)
</pre>

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This project represents the culmination of completing the Boot.dev course on Retrieval-Augmented Generation (RAG) systems. Built entirely from scratch, it demonstrates deep understanding of core RAG concepts through custom implementations of search algorithms, embedding techniques, and LLM integrations—without relying on high-level libraries for the fundamental logic.

## 🎯 Topics Covered

### Keyword Search
Implemented a complete keyword-based search system using an inverted index data structure. The system tokenizes text, removes stopwords, applies stemming, and builds an index mapping terms to document IDs for efficient lookups.

**Key Components:**
- **Inverted Index**: Custom-built index with term-to-document mappings, term frequency tracking, and document length storage
- **Text Processing**: Lowercasing, punctuation removal, stopword filtering, and Porter stemming
- **Search Logic**: Boolean AND operations across query terms with result deduplication

### TF-IDF (Term Frequency-Inverse Document Frequency)
Developed a custom TF-IDF scoring implementation that calculates term importance within documents and across the corpus.

**Implementation Details:**
- **Term Frequency (TF)**: Raw count of term occurrences in a document
- **Inverse Document Frequency (IDF)**: Logarithmic scaling based on document frequency: `log((N+1)/(df+1))`
- **TF-IDF Score**: Product of TF and IDF for relevance ranking

### BM25 (Best Matching 25)
Built a full BM25 ranking algorithm from scratch, including both TF and IDF variants with configurable parameters.

**Custom Implementation:**
- **BM25 TF**: Saturation-based term frequency with length normalization: `(tf * (k1 + 1)) / (tf + k1 * length_norm)`
- **BM25 IDF**: Probabilistic IDF with smoothing: `log((N - df + 0.5) / (df + 0.5) + 1)`
- **Parameters**: Tunable k1 (1.5) for term saturation and b (0.75) for length normalization
- **Search**: Combines BM25 scores across query terms for document ranking

### Chunking
Implemented two chunking strategies for handling long documents in semantic search.

**Strategies:**
- **Fixed Chunking**: Divides text into equal-sized chunks with configurable overlap
- **Semantic Chunking**: Splits text at sentence boundaries with sentence-based overlap, preserving semantic coherence

### Embeddings
Created a custom embedding pipeline using SentenceTransformers for text vectorization.

**Features:**
- **Model Integration**: Uses all-MiniLM-L6-v2 for 384-dimensional embeddings
- **Caching System**: Persistent storage of document embeddings for performance
- **Batch Processing**: Efficient encoding of multiple texts with progress tracking

### Vector Search
Developed semantic search using cosine similarity for vector-based document retrieval.

**Implementation:**
- **Cosine Similarity**: Custom calculation: `dot(a,b) / (|a| * |b|)`
- **Ranking**: Sorts documents by similarity scores to query embedding
- **Chunked Search**: Searches across document chunks and aggregates scores by document

### LLMs (Large Language Models)
Integrated Google's Gemini 2.5 Flash Lite for various generative tasks.

**Uses:**
- **Query Enhancement**: Spelling correction, query rewriting, and expansion
- **Reranking**: Individual scoring, batch ranking, and cross-encoder reranking
- **Generation**: Response synthesis for RAG applications

### RAG
Implemented RAG with capabilities for intelligent query processing and result refinement.

**Agentic Features:**
- **Query Enhancement**: Three methods (spell, rewrite, expand) using LLM reasoning
- **Reranking Strategies**:
  - Individual: Per-document relevance scoring
  - Batch: Holistic ranking of result sets
  - Cross-Encoder: Fine-tuned relevance prediction
- **Hybrid Scoring**: Combines keyword and semantic scores with Reciprocal Rank Fusion (RRF)

## 🚀 Key Features

### Multimodal Search
Extended the system with CLIP-based image search capabilities.

**Implementation:**
- **Image Embeddings**: Uses CLIP-ViT-B-32 for joint text-image embedding space
- **Cross-Modal Retrieval**: Searches text documents using image queries
- **Similarity Matching**: Cosine similarity between image and text embeddings

### Hybrid Search
Combines keyword and semantic approaches for robust retrieval.

**Methods:**
- **Weighted Combination**: Linear interpolation of normalized BM25 and semantic scores
- **Reciprocal Rank Fusion (RRF)**: Rank-based fusion with configurable k parameter
- **Normalization**: Min-max scaling for score harmonization

### Retrieval-Augmented Generation (RAG)
Built multiple RAG variants for different use cases.

**Variants:**
- **Basic RAG**: Context-aware answer generation
- **Summarization**: Multi-document synthesis
- **Citations**: Source-attributed responses
- **Conversational Q&A**: Natural language question answering


## 🛠️ Usage

### Building the Index
```bash
python cli/keyword_search_cli.py build
```

### Keyword Search
```bash
# Basic search
python cli/keyword_search_cli.py search "action movie"

# Get TF-IDF scores
python cli/keyword_search_cli.py tfidf 1 "hero"

# BM25 search
python cli/keyword_search_cli.py bm25search "sci-fi adventure"
```

### Semantic Search
```bash
# Embed and search
python cli/semantic_search_cli.py search "space exploration" --limit 5

# Chunked search
python cli/semantic_search_cli.py search_chunked "alien invasion" --limit 3
```

### Hybrid Search
```bash
# Weighted search (50% keyword, 50% semantic)
python cli/hybrid_search_cli.py weighted-search "superhero movie" --alpha 0.5

# RRF search with enhancement
python cli/hybrid_search_cli.py rrf-search "bear horror" --enhance rewrite --rerank-method cross_encoder
```

### Multimodal Search
```bash
python cli/multimodal_search_cli.py image_search path/to/image.jpg
```

### RAG Applications
```bash
# Basic RAG
python cli/augmented_generation_cli.py rag "What are the best action movies?"

# Summarization
python cli/augmented_generation_cli.py summarize "space operas" --limit 10

# Citations
python cli/augmented_generation_cli.py citations "movies about artificial intelligence"

# Q&A
python cli/augmented_generation_cli.py question "Recommend a movie similar to Inception"
```

## 🔧 Custom Implementations

This project emphasizes building core algorithms from scratch to demonstrate deep understanding:

- **No off-the-shelf search libraries**: All indexing, scoring, and retrieval logic implemented manually
- **Custom similarity calculations**: Cosine similarity and normalization functions
- **Hand-rolled ranking algorithms**: BM25, TF-IDF, and RRF implementations
- **Purpose-built chunking**: Both fixed and semantic strategies without external chunkers
- **Integrated LLM workflows**: Direct API calls for enhancement and generation

The result is a comprehensive RAG system that showcases mastery of the underlying concepts while providing practical, extensible search capabilities.
