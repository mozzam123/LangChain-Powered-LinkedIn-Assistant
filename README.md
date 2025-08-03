# 🤖 LangChain-Powered LinkedIn Assistant (Groq Edition)

A sophisticated AI-powered assistant that leverages your LinkedIn data to provide intelligent insights and conversational interactions. Built with LangChain, FAISS vector storage, and powered by Groq's lightning-fast LLM inference.

## 🚀 Features

- **Smart Data Processing**: Automatically loads and processes LinkedIn connections and messages from CSV exports
- **Vector Search**: Uses FAISS for efficient semantic search across your LinkedIn data
- **Conversational Memory**: Maintains context across conversations using combined buffer and summary memory
- **Fast LLM Integration**: Powered by Groq's Llama3-8b model for rapid responses
- **Modular Architecture**: Clean, extensible codebase with separate modules for different functionalities

## 🏗️ Architecture

```
LangChain-Powered-LinkedIn-Assistant/
├── embedding/           # Embedding model management
├── loaders/            # CSV data loaders for LinkedIn exports
├── memory/             # Conversation memory management
├── splitter/           # Text chunking utilities
├── vectorstore/        # FAISS vector storage
├── main.py            # CLI interface
└── pipeline.py        # Data processing pipeline
```

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- LinkedIn data exports (CSV format)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LangChain-Powered-LinkedIn-Assistant.git
   cd LangChain-Powered-LinkedIn-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Prepare your LinkedIn data**
   - Export your LinkedIn connections to `data/connections.csv`
   - Export your LinkedIn messages to `data/messages.csv`

## 🚀 Usage

### 1. Build the Vector Index
First, process your LinkedIn data and build the FAISS vector index:

```bash
python pipeline.py
```

This will:
- Load your LinkedIn connections and messages
- Split documents into chunks
- Generate embeddings using sentence-transformers
- Build a FAISS vector index for fast retrieval

### 2. Start the Assistant
Run the interactive CLI assistant:

```bash
python main.py
```

Ask questions about your LinkedIn network, connections, or message history. The assistant will provide intelligent responses based on your data.

## 🔧 Configuration

### Memory Types
The assistant supports three memory configurations:
- **Buffer**: Stores recent conversation history
- **Summary**: Maintains conversation summaries
- **Combined**: Uses both buffer and summary (default)

### Embedding Model
Default: `sentence-transformers/all-MiniLM-L6-v2`
- Fast and efficient for semantic search
- Can be customized in `embedding/embedder.py`

### LLM Model
Default: `llama3-8b-8192` via Groq
- Optimized for speed and performance
- Configurable in `main.py`

## 📊 Data Format

### Connections CSV
Expected columns:
- `First Name`
- `Last Name`
- `Company`
- `Position`

### Messages CSV
Expected columns:
- `Content`
- `From`
- `To`
- `Date`

## 🧠 How It Works

1. **Data Loading**: CSV loaders parse LinkedIn exports into LangChain Document objects
2. **Text Splitting**: Documents are chunked for optimal embedding and retrieval
3. **Vectorization**: Text chunks are embedded using sentence transformers
4. **Indexing**: FAISS creates a searchable vector index
5. **Retrieval**: User queries are embedded and matched against stored vectors
6. **Generation**: Groq LLM generates contextual responses using retrieved information
7. **Memory**: Conversation history is maintained for context continuity

## 🔍 Example Queries

- "Who are my connections at Google?"
- "What conversations have I had about job opportunities?"
- "Show me people in my network who work in AI"
- "What was my last conversation with [person's name]?"

## 🛡️ Security & Privacy

- All data is processed locally
- No data is sent to external services except for LLM inference
- Vector index is stored locally in `vectorstore/faiss_index/`
- Environment variables keep API keys secure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the powerful framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Groq](https://groq.com/) for lightning-fast LLM inference
- [Hugging Face](https://huggingface.co/) for the embedding models

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the existing issues for solutions
- Review the code documentation

---

**Note**: This assistant processes your personal LinkedIn data. Ensure you have the right to use this data and follow LinkedIn's terms of service. 