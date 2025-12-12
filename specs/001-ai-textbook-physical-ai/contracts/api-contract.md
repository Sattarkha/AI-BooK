# API Contract: Physical AI & Humanoid Robotics Textbook

## Base URL
`https://api.textbook-physical-ai.com/v1`

## Authentication
Most endpoints require authentication via Bearer token:
```
Authorization: Bearer {token}
```
Anonymous access is allowed for basic content viewing.

## Endpoints

### Content Management

#### GET /modules
Retrieve all textbook modules
- **Auth**: Optional
- **Response**: 200 - Array of TextbookModule objects
- **Query Params**:
  - `active` (boolean, default: true) - Filter active modules
  - `limit` (integer) - Number of results to return
  - `offset` (integer) - Number of results to skip

#### GET /modules/{moduleId}
Retrieve a specific textbook module
- **Auth**: Optional
- **Response**: 200 - TextbookModule object
- **Errors**: 404 if module not found

#### GET /modules/{moduleId}/chapters
Retrieve chapters for a specific module
- **Auth**: Optional
- **Response**: 200 - Array of Chapter objects
- **Query Params**:
  - `active` (boolean, default: true) - Filter active chapters
  - `limit` (integer) - Number of results to return
  - `offset` (integer) - Number of results to skip

#### GET /chapters/{chapterId}
Retrieve a specific chapter
- **Auth**: Optional
- **Response**: 200 - Chapter object with full content
- **Errors**: 404 if chapter not found

#### GET /chapters/{chapterId}/exercises
Retrieve exercises for a specific chapter
- **Auth**: Optional
- **Response**: 200 - Array of Exercise objects
- **Query Params**:
  - `active` (boolean, default: true) - Filter active exercises

#### GET /chapters/{chapterId}/assessments
Retrieve assessments for a specific chapter
- **Auth**: Optional
- **Response**: 200 - Array of Assessment objects
- **Query Params**:
  - `active` (boolean, default: true) - Filter active assessments

### User Management

#### POST /auth/register
Register a new user
- **Auth**: None
- **Body**:
  ```json
  {
    "email": "user@example.com",
    "name": "User Name",
    "password": "securePassword123"
  }
  ```
- **Response**: 201 - User object with access token
- **Errors**: 400 for invalid input, 409 for duplicate email

#### POST /auth/login
Login existing user
- **Auth**: None
- **Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "securePassword123"
  }
  ```
- **Response**: 200 - User object with access token
- **Errors**: 401 for invalid credentials

#### GET /users/profile
Get current user profile
- **Auth**: Required
- **Response**: 200 - User object
- **Errors**: 401 for invalid token

#### PUT /users/profile
Update user profile
- **Auth**: Required
- **Body**: Partial User object
- **Response**: 200 - Updated User object
- **Errors**: 400 for invalid input

### Progress Tracking

#### GET /users/progress
Get user's progress across all chapters
- **Auth**: Required
- **Response**: 200 - Array of UserProgress objects
- **Query Params**:
  - `moduleId` (UUID) - Filter by specific module
  - `chapterId` (UUID) - Filter by specific chapter

#### POST /users/progress/{chapterId}
Update progress for a specific chapter
- **Auth**: Required
- **Body**:
  ```json
  {
    "status": "in-progress",
    "completionPercentage": 0.75,
    "timeSpent": 1800
  }
  ```
- **Response**: 200 - Updated UserProgress object
- **Errors**: 404 if chapter not found

#### GET /users/progress/{chapterId}/assessment
Get user's assessment result for a chapter
- **Auth**: Required
- **Response**: 200 - Assessment result
- **Errors**: 404 if assessment not found

#### POST /users/progress/{chapterId}/assessment
Submit assessment for a chapter
- **Auth**: Required
- **Body**: Assessment answers
- **Response**: 200 - Assessment result with score
- **Errors**: 400 for invalid answers, 404 if assessment not found

### RAG Chatbot

#### POST /chat/sessions
Create a new chat session
- **Auth**: Optional (for anonymous users)
- **Body** (optional):
  ```json
  {
    "sessionId": "existing-session-id" // for continuing existing session
  }
  ```
- **Response**: 201 - ChatSession object with new session ID

#### POST /chat/sessions/{sessionId}/messages
Send a message to the chatbot
- **Auth**: Optional
- **Body**:
  ```json
  {
    "content": "What is ROS 2?",
    "selectedText": "ROS 2 is a set of software libraries and tools that help you build robot applications" // optional
  }
  ```
- **Response**: 200 - Array of ChatMessage objects (including the response)
- **Errors**: 429 if rate limit exceeded

#### GET /chat/sessions/{sessionId}/messages
Retrieve chat history
- **Auth**: Optional
- **Response**: 200 - Array of ChatMessage objects
- **Query Params**:
  - `limit` (integer, default: 50) - Number of messages to return
  - `offset` (integer, default: 0) - Number of messages to skip

### Content Search

#### GET /search
Search across textbook content
- **Auth**: Optional
- **Response**: 200 - Array of search results
- **Query Params**:
  - `q` (string, required) - Search query
  - `type` (string) - Filter by content type (chapter, exercise, code-example)
  - `moduleId` (UUID) - Limit search to specific module
  - `limit` (integer, default: 10) - Number of results to return

### Translation

#### GET /translation/languages
Get available languages
- **Auth**: Optional
- **Response**: 200 - Array of available languages with codes

#### POST /translation/toggle
Toggle content language preference
- **Auth**: Required
- **Body**:
  ```json
  {
    "languageCode": "ur" // e.g., "en", "ur"
  }
  ```
- **Response**: 200 - Success message

### Simulation Exercises

#### GET /simulations
Get available simulation exercises
- **Auth**: Optional
- **Response**: 200 - Array of SimulationExercise objects
- **Query Params**:
  - `moduleId` (UUID) - Filter by module
  - `environment` (string) - Filter by simulation environment

#### GET /simulations/{simulationId}
Get specific simulation exercise
- **Auth**: Optional
- **Response**: 200 - SimulationExercise object
- **Errors**: 404 if simulation not found

## Error Responses

All error responses follow this format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {} // optional additional details
  }
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Request body failed validation
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `UNAUTHORIZED`: Authentication required or failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server-side error occurred

## Rate Limits
- Unauthenticated requests: 100/hour per IP
- Authenticated requests: 1000/hour per user
- Chat endpoints: 10/minute per session