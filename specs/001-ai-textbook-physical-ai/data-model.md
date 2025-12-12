# Data Model: Physical AI & Humanoid Robotics Textbook

## Entity Models

### TextbookModule
- **ID**: UUID (Primary Key)
- **name**: String (e.g., "ROS 2", "Digital Twin", "AI-Robot Brain", "VLA")
- **title**: String (display title)
- **description**: Text (brief description of the module)
- **order**: Integer (sequence number for navigation)
- **learningObjectives**: Array<String> (list of learning objectives)
- **prerequisites**: Array<String> (required knowledge before starting)
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### Chapter
- **ID**: UUID (Primary Key)
- **moduleId**: UUID (Foreign Key to TextbookModule)
- **title**: String (chapter title)
- **slug**: String (URL-friendly identifier)
- **content**: Text (markdown content)
- **order**: Integer (sequence within module)
- **estimatedReadingTime**: Integer (in minutes)
- **learningObjectives**: Array<String>
- **prerequisites**: Array<String>
- **exercises**: Array<UUID> (references to Exercise entities)
- **assessments**: Array<UUID> (references to Assessment entities)
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### Exercise
- **ID**: UUID (Primary Key)
- **chapterId**: UUID (Foreign Key to Chapter)
- **title**: String
- **description**: Text
- **type**: String (e.g., "coding", "simulation", "theory", "multiple-choice")
- **difficulty**: String (e.g., "beginner", "intermediate", "advanced")
- **content**: Text (markdown with instructions)
- **solution**: Text (markdown with solution)
- **hints**: Array<String> (progressive hints)
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### Assessment
- **ID**: UUID (Primary Key)
- **chapterId**: UUID (Foreign Key to Chapter)
- **title**: String
- **description**: Text
- **type**: String (e.g., "quiz", "project", "simulation")
- **maxScore**: Integer
- **passingScore**: Integer
- **content**: Text (markdown with questions/tasks)
- **solution**: Text (markdown with solution)
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### User
- **ID**: UUID (Primary Key)
- **email**: String (unique, required for registered users)
- **name**: String
- **userType**: String (e.g., "student", "educator", "guest")
- **learningPreferences**: JSON (personalization settings)
- **languagePreference**: String (default: "en", options: ["en", "ur", ...])
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### UserProgress
- **ID**: UUID (Primary Key)
- **userId**: UUID (Foreign Key to User)
- **chapterId**: UUID (Foreign Key to Chapter)
- **status**: String (e.g., "not-started", "in-progress", "completed")
- **completionPercentage**: Float (0.0 to 1.0)
- **lastAccessed**: DateTime
- **timeSpent**: Integer (in seconds)
- **score**: Integer (for chapters with assessments)
- **createdAt**: DateTime
- **updatedAt**: DateTime

### CodeExample
- **ID**: UUID (Primary Key)
- **chapterId**: UUID (Foreign Key to Chapter)
- **title**: String
- **description**: Text
- **language**: String (e.g., "python", "c++", "bash")
- **code**: Text (the actual code content)
- **outputExample**: Text (expected output or behavior)
- **simulationLink**: String (optional link to simulation)
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

### RAGKnowledgeBase
- **ID**: UUID (Primary Key)
- **sourceType**: String (e.g., "chapter", "exercise", "code-example")
- **sourceId**: UUID (ID of the source entity)
- **content**: Text (processed content for RAG)
- **embeddingVector**: Array<Float> (vector embedding of the content)
- **metadata**: JSON (additional metadata for retrieval)
- **createdAt**: DateTime
- **updatedAt**: DateTime

### ChatSession
- **ID**: UUID (Primary Key)
- **userId**: UUID (Foreign Key to User, nullable for anonymous)
- **sessionId**: String (for anonymous users)
- **createdAt**: DateTime
- **lastInteraction**: DateTime
- **isActive**: Boolean

### ChatMessage
- **ID**: UUID (Primary Key)
- **sessionId**: UUID (Foreign Key to ChatSession)
- **sender**: String (e.g., "user", "assistant")
- **content**: Text (message content)
- **timestamp**: DateTime
- **sourceContext**: JSON (relevant textbook content used for response)
- **isFollowUp**: Boolean (indicates if this is a follow-up to previous question)

### SimulationExercise
- **ID**: UUID (Primary Key)
- **chapterId**: UUID (Foreign Key to Chapter)
- **title**: String
- **description**: Text
- **simulationEnvironment**: String (e.g., "gazebo", "isaac-sim", "unity")
- **requirements**: Array<String> (hardware/software requirements)
- **instructions**: Text (step-by-step instructions)
- **expectedOutcome**: Text
- **createdAt**: DateTime
- **updatedAt**: DateTime
- **isActive**: Boolean

## Relationships

### TextbookModule
- Has many: Chapter
- Has many: Exercise (through Chapter)
- Has many: Assessment (through Chapter)

### Chapter
- Belongs to: TextbookModule
- Has many: Exercise
- Has many: Assessment
- Has many: CodeExample
- Has many: SimulationExercise
- Has many: UserProgress (one per user)

### User
- Has many: UserProgress
- Has many: ChatSession
- Has many: ChatMessage (through ChatSession)

### ChatSession
- Belongs to: User (optional)
- Has many: ChatMessage

## State Transitions

### UserProgress States
- `not-started` → `in-progress` (when user starts chapter)
- `in-progress` → `completed` (when user finishes chapter)
- `completed` → `in-progress` (if user wants to revisit)

### ChatSession States
- `active` → `inactive` (after period of inactivity or explicit end)

## Validation Rules

### TextbookModule
- Name must be unique
- Order must be positive integer
- Must have at least one chapter

### Chapter
- Title and slug must be unique within module
- Order must be positive integer
- Content must be valid markdown

### User
- Email must be unique if provided
- Learning preferences must follow defined schema

### UserProgress
- Completion percentage must be between 0.0 and 1.0
- Score must be between 0 and maxScore (if assessment exists)

### RAGKnowledgeBase
- Each sourceId can have multiple entries (for different content chunks)
- Embedding vectors must have consistent dimensions