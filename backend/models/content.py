from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TextbookModule(BaseModel):
    id: UUID4
    name: str  # e.g., "ROS 2", "Digital Twin", "AI-Robot Brain", "VLA"
    title: str
    description: str
    order: int  # sequence number for navigation
    learning_objectives: List[str]
    prerequisites: List[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class Chapter(BaseModel):
    id: UUID4
    module_id: UUID4  # Foreign Key to TextbookModule
    title: str
    slug: str  # URL-friendly identifier
    content: str  # markdown content
    order: int  # sequence within module
    estimated_reading_time: int  # in minutes
    learning_objectives: List[str]
    prerequisites: List[str]
    exercises: List[UUID4]  # references to Exercise entities
    assessments: List[UUID4]  # references to Assessment entities
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class ExerciseType(str, Enum):
    CODING = "coding"
    SIMULATION = "simulation"
    THEORY = "theory"
    MULTIPLE_CHOICE = "multiple-choice"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Exercise(BaseModel):
    id: UUID4
    chapter_id: UUID4  # Foreign Key to Chapter
    title: str
    description: str
    type: ExerciseType
    difficulty: DifficultyLevel
    content: str  # markdown with instructions
    solution: Optional[str]  # markdown with solution
    hints: List[str]  # progressive hints
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class AssessmentType(str, Enum):
    QUIZ = "quiz"
    PROJECT = "project"
    SIMULATION = "simulation"


class Assessment(BaseModel):
    id: UUID4
    chapter_id: UUID4  # Foreign Key to Chapter
    title: str
    description: str
    type: AssessmentType
    max_score: int
    passing_score: int
    content: str  # markdown with questions/tasks
    solution: Optional[str]  # markdown with solution
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class UserType(str, Enum):
    STUDENT = "student"
    EDUCATOR = "educator"
    GUEST = "guest"


class User(BaseModel):
    id: UUID4
    email: Optional[str]  # unique, required for registered users
    name: Optional[str]
    user_type: UserType
    learning_preferences: Optional[Dict[str, Any]]  # personalization settings
    language_preference: str = "en"  # default: "en", options: ["en", "ur", ...]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class UserProgressStatus(str, Enum):
    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"


class UserProgress(BaseModel):
    id: UUID4
    user_id: UUID4  # Foreign Key to User
    chapter_id: UUID4  # Foreign Key to Chapter
    status: UserProgressStatus
    completion_percentage: float  # 0.0 to 1.0
    last_accessed: datetime
    time_spent: int  # in seconds
    score: Optional[int]  # for chapters with assessments
    created_at: datetime
    updated_at: datetime


class CodeExample(BaseModel):
    id: UUID4
    chapter_id: UUID4  # Foreign Key to Chapter
    title: str
    description: str
    language: str  # e.g., "python", "c++", "bash"
    code: str  # the actual code content
    output_example: Optional[str]  # expected output or behavior
    simulation_link: Optional[str]  # optional link to simulation
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class RAGSourceType(str, Enum):
    CHAPTER = "chapter"
    EXERCISE = "exercise"
    CODE_EXAMPLE = "code-example"


class RAGKnowledgeBase(BaseModel):
    id: UUID4
    source_type: RAGSourceType
    source_id: UUID4  # ID of the source entity
    content: str  # processed content for RAG
    embedding_vector: List[float]  # vector embedding of the content
    metadata: Dict[str, Any]  # additional metadata for retrieval
    created_at: datetime
    updated_at: datetime


class ChatSession(BaseModel):
    id: UUID4
    user_id: Optional[UUID4]  # Foreign Key to User, nullable for anonymous
    session_id: str  # for anonymous users
    created_at: datetime
    last_interaction: datetime
    is_active: bool = True


class ChatMessageSender(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    id: UUID4
    session_id: UUID4  # Foreign Key to ChatSession
    sender: ChatMessageSender
    content: str  # message content
    timestamp: datetime
    source_context: Optional[Dict[str, Any]]  # relevant textbook content used for response
    is_follow_up: bool = False  # indicates if this is a follow-up to previous question


class SimulationExercise(BaseModel):
    id: UUID4
    chapter_id: UUID4  # Foreign Key to Chapter
    title: str
    description: str
    simulation_environment: str  # e.g., "gazebo", "isaac-sim", "unity"
    requirements: List[str]  # hardware/software requirements
    instructions: str  # step-by-step instructions
    expected_outcome: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True