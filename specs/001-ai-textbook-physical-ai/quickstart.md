# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js 18+ (for Docusaurus frontend)
- Python 3.11+ (for backend services)
- Git
- Access to OpenAI API (for RAG functionality)
- Docker (optional, for local development)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-textbook-physical-ai
```

### 2. Frontend Setup (Docusaurus)

```bash
# Navigate to project root
cd E:\agentic-AI\prompt and context engineering\AI-Book\AI-Book  # Update path as needed

# Install frontend dependencies
npm install

# Start the development server
npm start
```

The textbook will be available at `http://localhost:3000`

### 3. Backend Setup (FastAPI)

```bash
# In a separate terminal, navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export QDRANT_URL="http://localhost:6333"  # Or your Qdrant instance URL

# Start the backend server
uvicorn main:app --reload --port 8000
```

### 4. Vector Database Setup (Qdrant)

```bash
# Option 1: Docker (recommended for development)
docker run -d --name qdrant-container -p 6333:6333 qdrant/qdrant

# Option 2: Local installation
# Follow instructions at: https://qdrant.tech/documentation/quick-start/
```

## Content Development

### Adding New Chapters

1. Create a new markdown file in the `docs/` directory
2. Add the content using Docusaurus markdown syntax
3. Update the `sidebars.js` file to include the new chapter in the navigation
4. For interactive elements, create React components in `src/components/`

### Running Content Validation

```bash
# Validate all content for broken links and formatting
npm run build

# Run tests
npm test
```

## Building for Production

### Static Site Build

```bash
# Build the static site
npm run build

# Serve locally to test
npm run serve
```

### Backend Deployment

1. Set production environment variables
2. Deploy the FastAPI application to your preferred cloud provider
3. Ensure the Qdrant vector database is accessible from your deployed application

## Key Features Setup

### RAG Chatbot Configuration

The RAG system requires:
- Textbook content to be processed into vector embeddings
- OpenAI API access for generating responses
- Qdrant database for storing and retrieving embeddings

### Personalization System

User preferences and personalization features are managed through:
- Local storage for basic preferences
- Backend user accounts (optional, for advanced features)

### Urdu Translation Toggle

The translation system works by:
- Pre-translated content stored in parallel structures
- JavaScript toggle functionality to switch between languages
- Preserved formatting and interactive elements

## Testing

### Frontend Tests

```bash
npm test
npm run test:e2e  # End-to-end tests
```

### Backend Tests

```bash
cd backend
python -m pytest tests/
```

## Deployment

### GitHub Pages (Static Content)

The static textbook content is deployed to GitHub Pages using GitHub Actions. Push changes to the main branch to trigger deployment.

### Backend Services

Deploy the FastAPI backend to your preferred cloud provider (AWS, Azure, GCP, or Vercel/Netlify with serverless functions).

## Troubleshooting

### Common Issues

1. **Content not updating**: Clear browser cache and restart development server
2. **RAG chatbot not responding**: Check that the backend server is running and API keys are set
3. **Slow performance**: Ensure you have sufficient local resources and consider using production builds

### Development Tips

- Use the development server for real-time content editing
- Test the RAG functionality with simple queries first
- Validate content changes before committing