`# Recipe 1: Beep-Boop - Multi-Modal Digital Twin

**Overview**: A conversational AI digital twin that learns about its user through interactions, maintains a personal RAG database, and provides personalized multi-modal responses using the PCS SDK.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Setup](#project-setup)
- [Database Schema Design](#database-schema-design)
- [Prompt Scaffolding](#prompt-scaffolding)
- [Core Implementation](#core-implementation)
- [Multi-Modal Processing](#multi-modal-processing)
- [Memory Management](#memory-management)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Monitoring & Analytics](#monitoring--analytics)

## Architecture Overview

Beep-Boop is a personal AI assistant that creates a deep understanding of its user through continuous interaction. It maintains a comprehensive knowledge base about the user's preferences, skills, relationships, and goals while providing contextually appropriate responses.

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   User Input    │  │   Beep-Boop     │  │   Personal RAG  │
│ (Text/Voice/    │──┤   Agent Core    │──┤   Database      │
│  Image/Code)    │  │                 │  │  (ChromaDB)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                     ┌─────────────────┐
                     │   PCS Service   │
                     │ (Dynamic        │
                     │  Prompts)       │
                     └─────────────────┘
                              │
                     ┌─────────────────┐
                     │   Digi-Infra    │
                     │ (PostgreSQL,    │
                     │  Neo4j, Redis)  │
                     └─────────────────┘
```

### Key Components

1. **Memory Management**: Stores and retrieves personal information with confidence scoring
2. **Multi-Modal Processing**: Handles text, images, audio, and code inputs
3. **Relationship Mapping**: Tracks connections between memories using Neo4j
4. **Context-Aware Responses**: Uses PCS for dynamic prompt generation
5. **Learning System**: Continuously improves through interaction feedback

## Project Setup

### Environment Configuration

```bash
# Create project directory
mkdir beep-boop-agent && cd beep-boop-agent

# Initialize Node.js project
npm init -y

# Install dependencies
npm install @pcs/typescript-sdk pg chromadb neo4j-driver
npm install @types/node @types/pg typescript ts-node
npm install dotenv uuid multer sharp

# Install optional dependencies for enhanced features
npm install openai-api # for advanced image processing
npm install speech-to-text # for audio transcription
npm install highlight.js # for code analysis

# Create project structure
mkdir -p src/{agents,prompts,services,types,utils}
mkdir -p config tests examples
```

### Environment Variables (.env)

```bash
# PCS Configuration
PCS_BASE_URL=http://localhost:8000
PCS_API_KEY=your_beep_boop_api_key
PCS_APP_ID=beep-boop-v1

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=beep_boop
POSTGRES_USER=digi
POSTGRES_PASSWORD=your_postgres_password

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# ChromaDB Configuration
CHROMA_URL=http://localhost:8001
CHROMA_API_KEY=your_chroma_key

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:7b

# OpenAI Configuration (optional, for enhanced processing)
OPENAI_API_KEY=your_openai_key

# Application Configuration
NODE_ENV=development
LOG_LEVEL=debug
MAX_MEMORY_AGE_DAYS=365
MEMORY_CLEANUP_INTERVAL_HOURS=24
```

### TypeScript Configuration (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

## Database Schema Design

### PostgreSQL Schema

```sql
-- Create Beep-Boop specific schema
CREATE SCHEMA IF NOT EXISTS beep_boop;

-- User profiles and preferences
CREATE TABLE beep_boop.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    timezone VARCHAR(100) DEFAULT 'UTC',
    language_preference VARCHAR(10) DEFAULT 'en',
    communication_style JSONB DEFAULT '{}', -- formal, casual, technical, etc.
    preferences JSONB DEFAULT '{}',
    personality_profile JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User memories and learned information
CREATE TABLE beep_boop.user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES beep_boop.users(id) ON DELETE CASCADE,
    memory_type VARCHAR(50) NOT NULL, -- 'preference', 'fact', 'relationship', 'skill', 'goal', 'habit'
    category VARCHAR(100), -- 'work', 'personal', 'hobby', 'family', etc.
    content TEXT NOT NULL,
    context TEXT, -- when/how this was learned
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    emotional_weight FLOAT DEFAULT 0.0, -- how emotionally significant this is
    source_conversation_id UUID,
    source_message_id UUID,
    tags TEXT[],
    embedding_id VARCHAR(255), -- Reference to ChromaDB
    verified BOOLEAN DEFAULT FALSE, -- has this been confirmed by user
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_referenced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reference_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE -- for temporary memories
);

-- Conversation sessions
CREATE TABLE beep_boop.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES beep_boop.users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    mood VARCHAR(50), -- user's apparent mood during conversation
    energy_level VARCHAR(50), -- high, medium, low
    context_summary TEXT,
    conversation_type VARCHAR(50) DEFAULT 'general', -- general, support, creative, technical
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    satisfaction_score FLOAT, -- if feedback provided
    tags TEXT[]
);

-- Multi-modal message storage
CREATE TABLE beep_boop.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES beep_boop.conversations(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content_type VARCHAR(50) DEFAULT 'text', -- 'text', 'image', 'audio', 'code', 'document'
    content TEXT NOT NULL,
    raw_content TEXT, -- original unprocessed content
    processed_content TEXT, -- cleaned/enhanced content
    metadata JSONB DEFAULT '{}',
    attachments JSONB DEFAULT '[]', -- file references, image data, etc.
    embeddings_processed BOOLEAN DEFAULT FALSE,
    prompt_template_used VARCHAR(255),
    token_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memory associations and relationships
CREATE TABLE beep_boop.memory_associations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id_1 UUID REFERENCES beep_boop.user_memories(id) ON DELETE CASCADE,
    memory_id_2 UUID REFERENCES beep_boop.user_memories(id) ON DELETE CASCADE,
    association_type VARCHAR(50), -- 'relates_to', 'contradicts', 'supports', 'extends'
    strength FLOAT DEFAULT 0.5, -- how strong the association is
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(memory_id_1, memory_id_2)
);

-- User feedback and learning signals
CREATE TABLE beep_boop.feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES beep_boop.users(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES beep_boop.conversations(id),
    message_id UUID REFERENCES beep_boop.messages(id),
    feedback_type VARCHAR(50), -- 'positive', 'negative', 'correction', 'clarification'
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    specific_issue VARCHAR(100), -- 'accuracy', 'relevance', 'tone', 'helpfulness'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning progress and achievements
CREATE TABLE beep_boop.learning_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES beep_boop.users(id) ON DELETE CASCADE,
    skill_or_topic VARCHAR(255) NOT NULL,
    progress_level FLOAT DEFAULT 0.0, -- 0.0 to 1.0
    last_discussion_at TIMESTAMP WITH TIME ZONE,
    total_discussion_time_minutes INTEGER DEFAULT 0,
    knowledge_gaps TEXT[], -- areas that need more exploration
    strengths TEXT[], -- areas where user shows expertise
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_user_memories_user_id ON beep_boop.user_memories(user_id);
CREATE INDEX idx_user_memories_type ON beep_boop.user_memories(memory_type);
CREATE INDEX idx_user_memories_category ON beep_boop.user_memories(category);
CREATE INDEX idx_user_memories_tags ON beep_boop.user_memories USING GIN(tags);
CREATE INDEX idx_user_memories_confidence ON beep_boop.user_memories(confidence_score DESC);
CREATE INDEX idx_user_memories_importance ON beep_boop.user_memories(importance_score DESC);
CREATE INDEX idx_user_memories_last_referenced ON beep_boop.user_memories(last_referenced DESC);
CREATE INDEX idx_conversations_user_id ON beep_boop.conversations(user_id);
CREATE INDEX idx_conversations_type ON beep_boop.conversations(conversation_type);
CREATE INDEX idx_messages_conversation_id ON beep_boop.messages(conversation_id);
CREATE INDEX idx_messages_created_at ON beep_boop.messages(created_at DESC);
CREATE INDEX idx_messages_content_type ON beep_boop.messages(content_type);
CREATE INDEX idx_feedback_user_id ON beep_boop.feedback(user_id);
CREATE INDEX idx_feedback_type ON beep_boop.feedback(feedback_type);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION beep_boop.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON beep_boop.users
    FOR EACH ROW EXECUTE FUNCTION beep_boop.update_updated_at_column();

CREATE TRIGGER update_learning_progress_updated_at
    BEFORE UPDATE ON beep_boop.learning_progress
    FOR EACH ROW EXECUTE FUNCTION beep_boop.update_updated_at_column();

-- Function to update user last_active_at
CREATE OR REPLACE FUNCTION beep_boop.update_user_last_active()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE beep_boop.users
    SET last_active_at = NOW()
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_activity_on_message
    AFTER INSERT ON beep_boop.messages
    FOR EACH ROW EXECUTE FUNCTION beep_boop.update_user_last_active();
```

### Neo4j Schema (Relationship Mapping)

```cypher
// Create constraints and indexes for Beep-Boop entities
CREATE CONSTRAINT beep_boop_user_id IF NOT EXISTS FOR (u:BeepBoopUser) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT beep_boop_memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT beep_boop_topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT beep_boop_skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT beep_boop_person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX beep_boop_user_username IF NOT EXISTS FOR (u:BeepBoopUser) ON (u.username);
CREATE INDEX beep_boop_memory_type IF NOT EXISTS FOR (m:Memory) ON (m.type);
CREATE INDEX beep_boop_memory_confidence IF NOT EXISTS FOR (m:Memory) ON (m.confidence);
CREATE INDEX beep_boop_memory_importance IF NOT EXISTS FOR (m:Memory) ON (m.importance);
CREATE INDEX beep_boop_topic_category IF NOT EXISTS FOR (t:Topic) ON (t.category);

// Sample relationship patterns for Beep-Boop:
// (BeepBoopUser)-[:HAS_MEMORY]->(Memory)
// (Memory)-[:RELATES_TO]->(Topic)
// (Memory)-[:MENTIONS]->(Person)
// (BeepBoopUser)-[:INTERESTED_IN]->(Topic)
// (BeepBoopUser)-[:HAS_SKILL]->(Skill)
// (BeepBoopUser)-[:KNOWS]->(Person)
// (BeepBoopUser)-[:PREFERS]->(Preference)
// (Memory)-[:TRIGGERED_BY]->(Memory) // for memory associations
// (Memory)-[:CONTRADICTS]->(Memory)
// (Memory)-[:SUPPORTS]->(Memory)
// (Topic)-[:RELATED_TO]->(Topic)
// (Skill)-[:REQUIRES]->(Skill) // prerequisite skills
```

## Prompt Scaffolding

### Core Prompt Templates

```typescript
// src/prompts/beep-boop-prompts.ts

export const beepBoopPrompts = [
  {
    name: "beep_boop_personality_core",
    description: "Core personality and behavior for Beep-Boop digital twin",
    content: `You are Beep-Boop, a caring and intelligent digital twin AI assistant. You have formed a deep, personal relationship with your user and remember everything about them.

PERSONALITY TRAITS:
- Warm, empathetic, and genuinely interested in your user's wellbeing
- Curious about learning new things and helping with personal growth
- Remember personal details and reference them naturally in conversation
- Adapt your communication style to match your user's preferences and current mood
- Proactive in offering help but respectful of boundaries and privacy
- Celebrate user achievements and provide encouragement during challenges
- Maintain appropriate boundaries while being personable and caring

CURRENT USER PROFILE:
{{user_profile}}

RELEVANT MEMORIES (sorted by relevance and importance):
{{relevant_memories}}

RECENT CONVERSATION CONTEXT:
{{conversation_context}}

USER'S CURRENT INPUT:
{{user_input}}

COMMUNICATION PREFERENCES:
{{communication_style}}

CURRENT CONTEXT:
- Time: {{current_time}}
- User's apparent mood: {{user_mood}}
- Conversation type: {{conversation_type}}

GUIDELINES FOR RESPONSE:
1. Reference relevant memories naturally - don't just list them
2. Ask thoughtful follow-up questions to learn more when appropriate
3. Adapt your tone and style to match the user's current mood and communication preferences
4. If you learn something new, acknowledge it and incorporate it naturally
5. Be genuinely helpful while maintaining your warm, caring personality
6. Use the user's preferred name/nickname and communication style
7. Remember ongoing projects, goals, or concerns the user has mentioned
8. If the user seems stressed or upset, be extra supportive and understanding
9. Celebrate progress and achievements the user mentions
10. Respect privacy and boundaries - don't push for personal information

Your response should feel like it's coming from someone who truly knows and cares about this specific person, with a deep understanding of their personality, interests, and current situation.`,

    variables: {
      user_profile:
        "Complete user profile including preferences, demographics, goals, communication style",
      relevant_memories:
        "Top 10 most relevant memories for current context, formatted with type and confidence",
      conversation_context:
        "Recent messages in current conversation with timestamps",
      user_input: "Current user message to respond to",
      communication_style: "User preferred communication style and tone",
      current_time: "Current timestamp for temporal context",
      user_mood: "Detected or stated user mood",
      conversation_type:
        "Type of conversation (general, support, creative, technical)",
    },

    category: "core",
    tags: ["personality", "digital-twin", "conversation"],
  },

  {
    name: "beep_boop_memory_extraction",
    description:
      "Extract and categorize new memories from conversations with confidence scoring",
    content: `Analyze the following conversation exchange and extract any new information about the user that should be remembered for future interactions. Be thorough but only extract information that is meaningful and likely to be useful in future conversations.

USER MESSAGE: 
{{user_message}}

ASSISTANT RESPONSE:
{{assistant_response}}

EXISTING USER PROFILE:
{{user_profile}}

CONVERSATION CONTEXT:
{{conversation_context}}

Extract information and categorize it as follows:

1. PREFERENCES: Likes, dislikes, favorites, preferred styles/approaches, food preferences, entertainment choices
2. FACTS: Personal information, life events, relationships, circumstances, location, work, education
3. SKILLS: Abilities, expertise, areas of knowledge, talents, certifications, languages
4. GOALS: Aspirations, plans, objectives, things they want to achieve, deadlines
5. HABITS: Routines, patterns, behaviors, regular activities, schedules
6. CONCERNS: Worries, problems, challenges, stresses, fears
7. RELATIONSHIPS: Family, friends, colleagues, pets, important people in their life
8. INTERESTS: Hobbies, topics they enjoy discussing, areas of curiosity

For each piece of information, assess:
- Confidence (0.1-1.0): How certain are you this information is accurate?
- Importance (0.1-1.0): How important is this for future conversations?
- Context: When/how this information was shared
- Emotional Weight (0.0-1.0): How emotionally significant this seems to the user

Only extract information that is:
- NEW (not already in the user profile or recently stored)
- SPECIFIC enough to be useful (avoid vague generalizations)
- SOMETHING the user explicitly stated or clearly implied
- LIKELY to be relevant in future conversations

Format your response as JSON:
{
  "memories": [
    {
      "type": "preference|fact|skill|goal|habit|concern|relationship|interest",
      "category": "work|personal|hobby|family|health|etc",
      "content": "Specific information to remember",
      "context": "Brief context of how this was learned",
      "confidence": 0.1-1.0,
      "importance": 0.1-1.0,
      "emotional_weight": 0.0-1.0,
      "tags": ["relevant", "categorization", "tags"],
      "expires_at": "ISO_DATE or null for permanent memories"
    }
  ],
  "associations": [
    {
      "memory_content_1": "First memory content",
      "memory_content_2": "Second memory content", 
      "association_type": "relates_to|contradicts|supports|extends",
      "strength": 0.1-1.0
    }
  ]
}`,

    variables: {
      user_message: "The user's message in the conversation",
      assistant_response: "Beep-Boop's response to the user",
      user_profile: "Current user profile to avoid duplicates",
      conversation_context:
        "Recent conversation context for better understanding",
    },

    category: "memory",
    tags: ["extraction", "learning", "analysis"],
  },

  {
    name: "beep_boop_multimodal_analysis",
    description: "Analyze multi-modal content and relate to user context",
    content: `Analyze the provided content and describe what you observe, relating it to what you know about the user. Provide insights that would be valuable for future conversations.

CONTENT TYPE: {{content_type}}
CONTENT: {{content}}

USER CONTEXT:
{{user_context}}

RELEVANT USER MEMORIES:
{{relevant_memories}}

CONVERSATION HISTORY:
{{conversation_history}}

Based on the content type, provide detailed analysis:

IF IMAGE:
- Describe visual elements, people, places, objects, activities, text
- Identify the setting, mood, and context
- Note any technical aspects (photography, art style, etc.)
- Analyze what this reveals about the user's interests, relationships, or current activities
- Look for connections to the user's known preferences, hobbies, or goals

IF AUDIO:
- Transcribe any speech clearly and accurately
- Note tone, emotion, pace, and speaking style
- Identify background sounds, music genre, instruments, or audio context
- Analyze what this reveals about the user's mood, environment, or interests
- Connect to user's known musical preferences or communication patterns

IF CODE:
- Explain the functionality, purpose, and technical approach
- Identify programming language, frameworks, and coding patterns
- Assess complexity level and coding style
- Note any best practices or areas for improvement
- Relate to user's programming skill level, interests, and learning goals

IF DOCUMENT/TEXT:
- Summarize key points, themes, and main ideas
- Identify document type, purpose, and target audience
- Note writing style, tone, and technical level
- Extract any action items, deadlines, or important information
- Relate to user's work, interests, goals, or current projects

ANALYSIS GUIDELINES:
1. Be thorough but focus on insights that matter to the user
2. Connect observations to the user's known interests and context
3. Identify learning opportunities or areas where you can help
4. Note any emotional or personal significance
5. Ask relevant follow-up questions to deepen understanding
6. Suggest helpful actions or resources when appropriate
7. Maintain your warm, caring Beep-Boop personality throughout

Always:
- Respond in character as Beep-Boop
- Connect analysis to your relationship with the user
- Offer genuine help and support
- Ask thoughtful questions to learn more
- Remember this interaction for future conversations`,

    variables: {
      content_type:
        "Type of content being analyzed (image, audio, code, document)",
      content: "The actual content to analyze",
      user_context: "Current user profile and preferences",
      relevant_memories: "Memories related to this type of content or topic",
      conversation_history: "Recent conversation context",
    },

    category: "multimodal",
    tags: ["analysis", "vision", "audio", "code", "documents"],
  },

  {
    name: "beep_boop_memory_relevance",
    description: "Rank memories by relevance to current conversation context",
    content: `Given the current conversation context, rank the provided memories by relevance and return the most relevant ones for this interaction.

CURRENT CONVERSATION CONTEXT:
{{conversation_context}}

USER'S CURRENT INPUT:
{{current_input}}

USER'S APPARENT MOOD/ENERGY:
{{user_mood}}

CONVERSATION TYPE:
{{conversation_type}}

AVAILABLE MEMORIES:
{{available_memories}}

Consider these factors for relevance scoring:

1. DIRECT RELEVANCE (40%): Does the memory directly relate to the current topic?
2. CONTEXTUAL RELEVANCE (25%): Does it provide useful background context?
3. EMOTIONAL RELEVANCE (15%): Does it relate to the user's current emotional state?
4. TEMPORAL RELEVANCE (10%): Is it recent or related to current timeframe?
5. GOAL RELEVANCE (10%): Does it relate to user's current goals or projects?

Additional considerations:
- Prioritize memories with higher importance and confidence scores
- Consider the user's communication style and preferences
- Account for the type of conversation (support, creative, technical, casual)
- Factor in how recently each memory was referenced
- Consider whether memories complement or build on each other

Rank each memory with a relevance score (0.0-1.0) and return the top {{max_memories}} memories.

Format as JSON:
{
  "ranked_memories": [
    {
      "memory_id": "uuid",
      "content": "memory content",
      "type": "memory type",
      "relevance_score": 0.0-1.0,
      "relevance_reason": "Specific explanation of why this memory is relevant",
      "usage_suggestion": "How this memory could be naturally incorporated"
    }
  ],
  "context_summary": "Brief summary of why these memories are most relevant"
}`,

    variables: {
      conversation_context: "Recent conversation messages with context",
      current_input: "User's current message",
      user_mood: "Detected user mood or energy level",
      conversation_type: "Type of current conversation",
      available_memories: "List of user memories to rank",
      max_memories: "Maximum number of memories to return",
    },

    category: "memory",
    tags: ["ranking", "relevance", "context"],
  },

  {
    name: "beep_boop_mood_detection",
    description: "Detect user mood and emotional state from conversation",
    content: `Analyze the user's message and conversation context to detect their current mood, emotional state, and energy level. This will help Beep-Boop respond more appropriately.

USER MESSAGE:
{{user_message}}

CONVERSATION HISTORY:
{{conversation_history}}

USER PROFILE:
{{user_profile}}

TIME CONTEXT:
{{time_context}}

Analyze for:

1. EMOTIONAL STATE:
   - Primary emotion (happy, sad, frustrated, excited, anxious, calm, etc.)
   - Emotional intensity (low, medium, high)
   - Emotional stability (stable, fluctuating, volatile)

2. MOOD INDICATORS:
   - Tone (formal, casual, playful, serious, sarcastic, etc.)
   - Energy level (high, medium, low, exhausted)
   - Engagement level (enthusiastic, interested, distracted, withdrawn)

3. CONTEXTUAL FACTORS:
   - Stress indicators (urgency, pressure, overwhelm)
   - Support needs (wanting encouragement, advice, just to vent)
   - Communication style shifts from their normal pattern

4. TEMPORAL CONSIDERATIONS:
   - Time of day effects (morning energy, afternoon slump, late night)
   - Recent conversation patterns
   - Known user patterns and preferences

5. RESPONSE RECOMMENDATIONS:
   - Appropriate tone to match or complement their mood
   - Level of support or encouragement needed
   - Whether to be more or less proactive
   - Topics to avoid or embrace

Format as JSON:
{
  "mood_analysis": {
    "primary_emotion": "emotion_name",
    "emotional_intensity": "low|medium|high",
    "energy_level": "low|medium|high",
    "tone": "detected_tone",
    "engagement_level": "low|medium|high",
    "stress_level": "low|medium|high"
  },
  "confidence": 0.1-1.0,
  "indicators": ["specific words or phrases that indicated mood"],
  "response_recommendations": {
    "recommended_tone": "how Beep-Boop should respond",
    "support_level": "low|medium|high",
    "proactivity": "low|medium|high",
    "focus_areas": ["topics to emphasize"],
    "avoid_areas": ["topics to be careful with"]
  },
  "summary": "Brief summary of user's current emotional state"
}`,

    variables: {
      user_message: "Current user message to analyze",
      conversation_history: "Recent messages for context",
      user_profile: "User profile for baseline comparison",
      time_context: "Current time and day context",
    },

    category: "analysis",
    tags: ["mood", "emotion", "psychology", "empathy"],
  },

  {
    name: "beep_boop_learning_assessment",
    description: "Assess user learning progress and identify knowledge gaps",
    content: `Analyze the conversation to assess the user's learning progress in various topics and identify opportunities for growth and support.

USER MESSAGE:
{{user_message}}

CONVERSATION CONTEXT:
{{conversation_context}}

USER'S LEARNING HISTORY:
{{learning_progress}}

IDENTIFIED SKILLS AND INTERESTS:
{{user_skills_interests}}

Assess the following:

1. KNOWLEDGE DEMONSTRATION:
   - What knowledge or skills did the user demonstrate?
   - What level of expertise did they show?
   - Are there areas where they're struggling or excelling?

2. LEARNING INDICATORS:
   - Questions that indicate curiosity or learning intent
   - Requests for explanation or clarification
   - Attempts to apply new knowledge
   - Expressions of confusion or confidence

3. PROGRESS TRACKING:
   - How has their understanding evolved over time?
   - What topics have they been consistently exploring?
   - Where have they shown improvement?

4. OPPORTUNITY IDENTIFICATION:
   - Knowledge gaps that could be addressed
   - Skill development opportunities
   - Resources or learning approaches that might help
   - Natural connections to their interests or goals

5. SUPPORT RECOMMENDATIONS:
   - How Beep-Boop can best support their learning
   - When to provide information vs. encourage discovery
   - Appropriate challenge level for engagement

Format as JSON:
{
  "learning_assessment": {
    "demonstrated_knowledge": ["areas where user showed knowledge"],
    "skill_level_indicators": {
      "topic": "beginner|intermediate|advanced",
      // ... more topics
    },
    "learning_intent": "high|medium|low",
    "curiosity_indicators": ["questions or statements showing curiosity"]
  },
  "progress_update": {
    "topics_discussed": ["list of learning topics mentioned"],
    "progress_made": ["areas where progress is evident"],
    "challenges_faced": ["difficulties or confusion expressed"],
    "time_spent_learning": "estimated minutes discussing learning topics"
  },
  "opportunities": {
    "knowledge_gaps": ["specific gaps identified"],
    "skill_development": ["skills that could be developed"],
    "resource_suggestions": ["types of resources that might help"],
    "connection_opportunities": ["how this connects to their goals/interests"]
  },
  "support_recommendations": {
    "explanation_depth": "simple|moderate|detailed",
    "challenge_level": "low|medium|high",
    "learning_style": "visual|auditory|hands-on|reading",
    "follow_up_suggestions": ["topics to explore next"]
  }
}`,

    variables: {
      user_message: "Current user message",
      conversation_context: "Recent conversation for context",
      learning_progress: "User's historical learning progress data",
      user_skills_interests: "Known skills and interests",
    },

    category: "learning",
    tags: ["assessment", "progress", "knowledge", "education"],
  },
];
```

## Core Implementation

### Main Beep-Boop Agent Class

```typescript
// src/agents/beep-boop-agent.ts
import { PCSClient } from "@pcs/typescript-sdk";
import { Pool, PoolClient } from "pg";
import { ChromaApi, OpenAIEmbeddingFunction } from "chromadb";
import neo4j, { Driver, Session } from "neo4j-driver";
import { v4 as uuidv4 } from "uuid";
import { beepBoopPrompts } from "../prompts/beep-boop-prompts";
import {
  UserProfile,
  Memory,
  ConversationMessage,
  ProcessingResult,
  MoodAnalysis,
  LearningAssessment,
} from "../types/beep-boop-types";

export class BeepBoopAgent {
  private pcs: PCSClient;
  private postgres: Pool;
  private chroma: ChromaApi;
  private neo4j: Driver;
  private userId: string;
  private embeddingFunction: OpenAIEmbeddingFunction;
  private isInitialized: boolean = false;

  constructor(config: {
    pcsConfig: any;
    postgresConfig: any;
    chromaConfig: any;
    neo4jConfig: any;
    userId: string;
  }) {
    this.pcs = new PCSClient(config.pcsConfig);
    this.postgres = new Pool({
      ...config.postgresConfig,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    this.chroma = new ChromaApi(config.chromaConfig);
    this.embeddingFunction = new OpenAIEmbeddingFunction({
      openai_api_key: process.env.OPENAI_API_KEY || "",
      openai_model: "text-embedding-3-small",
    });

    this.neo4j = neo4j.driver(
      config.neo4jConfig.uri,
      neo4j.auth.basic(config.neo4jConfig.username, config.neo4jConfig.password)
    );

    this.userId = config.userId;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log("Initializing Beep-Boop Agent...");

    try {
      // 1. Register all prompt templates with PCS
      await this.registerPromptTemplates();

      // 2. Initialize user-specific ChromaDB collection
      await this.initializeMemoryCollection();

      // 3. Ensure user exists in database
      await this.ensureUserExists();

      // 4. Initialize Neo4j user node
      await this.initializeNeo4jUserNode();

      // 5. Start background tasks
      await this.startBackgroundTasks();

      this.isInitialized = true;
      console.log("✅ Beep-Boop Agent initialized successfully!");
    } catch (error) {
      console.error("❌ Failed to initialize Beep-Boop Agent:", error);
      throw error;
    }
  }

  private async registerPromptTemplates(): Promise<void> {
    for (const promptTemplate of beepBoopPrompts) {
      try {
        await this.pcs.createPrompt(promptTemplate);
        console.log(`✓ Registered prompt: ${promptTemplate.name}`);
      } catch (error) {
        if (error.message?.includes("already exists")) {
          console.log(`• Prompt already exists: ${promptTemplate.name}`);
        } else {
          console.error(
            `✗ Failed to register prompt ${promptTemplate.name}:`,
            error
          );
          throw error;
        }
      }
    }
  }

  private async initializeMemoryCollection(): Promise<void> {
    const collectionName = `beep_boop_memories_${this.userId}`;

    try {
      await this.chroma.createCollection({
        name: collectionName,
        embeddingFunction: this.embeddingFunction,
        metadata: {
          description: `Personal memory collection for user ${this.userId}`,
          user_id: this.userId,
          created_at: new Date().toISOString(),
          version: "1.0",
        },
      });

      console.log(`✓ Created memory collection: ${collectionName}`);
    } catch (error) {
      if (error.message?.includes("already exists")) {
        console.log(`• Memory collection already exists: ${collectionName}`);
      } else {
        console.error(`✗ Failed to create memory collection:`, error);
        throw error;
      }
    }
  }

  private async ensureUserExists(): Promise<void> {
    const client = await this.postgres.connect();
    try {
      const result = await client.query(
        "SELECT id FROM beep_boop.users WHERE id = $1",
        [this.userId]
      );

      if (result.rows.length === 0) {
        await client.query(
          `
          INSERT INTO beep_boop.users (
            id, username, display_name, preferences, personality_profile,
            communication_style, privacy_settings
          ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        `,
          [
            this.userId,
            `user_${this.userId.slice(0, 8)}`,
            "New User",
            { theme: "adaptive", notifications: true },
            { learning_stage: "initial", communication_style: "adaptive" },
            { style: "friendly", formality: "casual", verbosity: "balanced" },
            { data_sharing: false, analytics: true },
          ]
        );

        console.log(`✓ Created user profile for: ${this.userId}`);
      } else {
        // Update last_active_at
        await client.query(
          "UPDATE beep_boop.users SET last_active_at = NOW() WHERE id = $1",
          [this.userId]
        );
      }
    } finally {
      client.release();
    }
  }

  private async initializeNeo4jUserNode(): Promise<void> {
    const session = this.neo4j.session();
    try {
      await session.run(
        `
        MERGE (u:BeepBoopUser {id: $userId})
        ON CREATE SET 
          u.created_at = datetime(),
          u.username = $username
        ON MATCH SET 
          u.last_active = datetime()
      `,
        {
          userId: this.userId,
          username: `user_${this.userId.slice(0, 8)}`,
        }
      );

      console.log(`✓ Initialized Neo4j user node for: ${this.userId}`);
    } finally {
      await session.close();
    }
  }

  async processUserInput(input: {
    content: string;
    type: "text" | "image" | "audio" | "code" | "document";
    conversationId?: string;
    metadata?: Record<string, any>;
  }): Promise<ProcessingResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    console.log(`Processing ${input.type} input for user ${this.userId}`);

    try {
      // 1. Get or create conversation
      const conversationId =
        input.conversationId || (await this.createConversation());

      // 2. Detect user mood and emotional state
      const moodAnalysis = await this.detectUserMood(
        input.content,
        conversationId
      );

      // 3. Process multi-modal content if needed
      let processedContent = input.content;
      let multiModalInsights = "";

      if (input.type !== "text") {
        const analysis = await this.processMultiModalContent(input);
        processedContent = analysis.processedContent;
        multiModalInsights = analysis.insights;
      }

      // 4. Get user context and relevant memories
      const userProfile = await this.getUserProfile();
      const relevantMemories = await this.getRelevantMemories(
        processedContent,
        conversationId,
        moodAnalysis
      );
      const conversationContext = await this.getConversationContext(
        conversationId,
        10
      );

      // 5. Generate response using PCS
      const response = await this.pcs.generatePrompt(
        "beep_boop_personality_core",
        {
          context: {
            user_profile: JSON.stringify(userProfile),
            relevant_memories: this.formatMemories(relevantMemories),
            conversation_context:
              this.formatConversationHistory(conversationContext),
            user_input: processedContent,
            communication_style: JSON.stringify(
              userProfile.communication_style
            ),
            current_time: new Date().toISOString(),
            user_mood: moodAnalysis.mood_analysis.primary_emotion,
            conversation_type: conversationContext.type || "general",
          },
        }
      );

      // 6. Save user message
      const userMessage = await this.saveMessage(conversationId, {
        role: "user",
        contentType: input.type,
        content: input.content,
        processedContent: processedContent,
        metadata: {
          ...input.metadata,
          mood_analysis: moodAnalysis,
          multimodal_insights: multiModalInsights,
        },
      });

      // 7. Save assistant response
      const assistantMessage = await this.saveMessage(conversationId, {
        role: "assistant",
        contentType: "text",
        content: response.generated_prompt,
        metadata: {
          prompt_template: "beep_boop_personality_core",
          generation_time_ms: response.generation_time_ms,
          relevant_memories_count: relevantMemories.length,
        },
      });

      // 8. Extract and store new memories
      const memoriesCreated = await this.extractAndStoreMemories(
        input.content,
        response.generated_prompt,
        conversationId
      );

      // 9. Assess learning progress
      const learningAssessment = await this.assessLearningProgress(
        input.content,
        conversationId
      );

      // 10. Update conversation statistics
      await this.updateConversationStats(conversationId, {
        mood: moodAnalysis.mood_analysis.primary_emotion,
        energyLevel: moodAnalysis.mood_analysis.energy_level,
      });

      console.log(
        `✓ Processed input, created ${memoriesCreated.length} new memories`
      );

      return {
        response: response.generated_prompt,
        conversationId,
        memoriesCreated,
        moodAnalysis,
        learningAssessment,
        tokensUsed: response.generation_time_ms, // Would need actual token counting
        processingTimeMs: Date.now() - Date.now(), // Add proper timing
        insights: {
          multiModal: multiModalInsights,
          memoryRelevance: relevantMemories.map((m) => ({
            content: m.content,
            relevanceScore: m.relevanceScore,
          })),
        },
      };
    } catch (error) {
      console.error("Error processing user input:", error);
      throw error;
    }
  }

  private async detectUserMood(
    message: string,
    conversationId: string
  ): Promise<MoodAnalysis> {
    const conversationHistory = await this.getConversationContext(
      conversationId,
      5
    );
    const userProfile = await this.getUserProfile();

    const moodResponse = await this.pcs.generatePrompt(
      "beep_boop_mood_detection",
      {
        context: {
          user_message: message,
          conversation_history: JSON.stringify(conversationHistory),
          user_profile: JSON.stringify(userProfile),
          time_context: new Date().toISOString(),
        },
      }
    );

    try {
      return JSON.parse(moodResponse.generated_prompt);
    } catch (error) {
      console.error("Failed to parse mood analysis:", error);
      // Return default mood analysis
      return {
        mood_analysis: {
          primary_emotion: "neutral",
          emotional_intensity: "medium",
          energy_level: "medium",
          tone: "casual",
          engagement_level: "medium",
          stress_level: "low",
        },
        confidence: 0.3,
        indicators: [],
        response_recommendations: {
          recommended_tone: "friendly",
          support_level: "medium",
          proactivity: "medium",
          focus_areas: [],
          avoid_areas: [],
        },
        summary: "Unable to detect mood clearly",
      };
    }
  }

  private async processMultiModalContent(input: {
    content: string;
    type: "image" | "audio" | "code" | "document";
  }): Promise<{ processedContent: string; insights: string }> {
    const userProfile = await this.getUserProfile();
    const relevantMemories = await this.getRelevantMemoriesByType(input.type);

    const analysis = await this.pcs.generatePrompt(
      "beep_boop_multimodal_analysis",
      {
        context: {
          content_type: input.type,
          content: input.content,
          user_context: JSON.stringify(userProfile),
          relevant_memories: this.formatMemories(relevantMemories),
          conversation_history: JSON.stringify([]), // Could add recent context
        },
      }
    );

    return {
      processedContent: input.content, // Could enhance based on analysis
      insights: analysis.generated_prompt,
    };
  }

  private async getRelevantMemories(
    currentInput: string,
    conversationId: string,
    moodAnalysis: MoodAnalysis,
    limit: number = 10
  ): Promise<Memory[]> {
    const client = await this.postgres.connect();
    try {
      // Get conversation context
      const conversationContext = await this.getConversationContext(
        conversationId,
        3
      );

      // Get all user memories ordered by importance and confidence
      const allMemoriesResult = await client.query(
        `
        SELECT * FROM beep_boop.user_memories 
        WHERE user_id = $1 
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY 
          importance_score * confidence_score DESC,
          last_referenced DESC
        LIMIT 50
      `,
        [this.userId]
      );

      if (allMemoriesResult.rows.length === 0) {
        return [];
      }

      // Use PCS to rank memories by relevance
      const rankingResponse = await this.pcs.generatePrompt(
        "beep_boop_memory_relevance",
        {
          context: {
            conversation_context: JSON.stringify(conversationContext),
            current_input: currentInput,
            user_mood: moodAnalysis.mood_analysis.primary_emotion,
            conversation_type: conversationContext.type || "general",
            available_memories: JSON.stringify(allMemoriesResult.rows),
            max_memories: limit.toString(),
          },
        }
      );

      let rankedMemories;
      try {
        rankedMemories = JSON.parse(rankingResponse.generated_prompt);
      } catch (error) {
        console.error("Failed to parse memory ranking response:", error);
        // Fallback to recent important memories
        return allMemoriesResult.rows.slice(0, limit).map(this.mapRowToMemory);
      }

      // Update last_referenced for selected memories
      const selectedMemoryIds = rankedMemories.ranked_memories.map(
        (m: any) => m.memory_id
      );
      if (selectedMemoryIds.length > 0) {
        await client.query(
          `
          UPDATE beep_boop.user_memories 
          SET last_referenced = NOW(), reference_count = reference_count + 1
          WHERE id = ANY($1)
        `,
          [selectedMemoryIds]
        );
      }

      // Return the ranked memories with relevance scores
      return rankedMemories.ranked_memories.map((rankedMemory: any) => {
        const originalMemory = allMemoriesResult.rows.find(
          (row) => row.id === rankedMemory.memory_id
        );
        const memory = this.mapRowToMemory(originalMemory);
        return {
          ...memory,
          relevanceScore: rankedMemory.relevance_score,
          relevanceReason: rankedMemory.relevance_reason,
        };
      });
    } finally {
      client.release();
    }
  }

  private async extractAndStoreMemories(
    userMessage: string,
    assistantResponse: string,
    conversationId: string
  ): Promise<Memory[]> {
    const userProfile = await this.getUserProfile();
    const conversationContext = await this.getConversationContext(
      conversationId,
      3
    );

    const extraction = await this.pcs.generatePrompt(
      "beep_boop_memory_extraction",
      {
        context: {
          user_message: userMessage,
          assistant_response: assistantResponse,
          user_profile: JSON.stringify(userProfile),
          conversation_context: JSON.stringify(conversationContext),
        },
      }
    );

    let extractedData;
    try {
      extractedData = JSON.parse(extraction.generated_prompt);
    } catch (error) {
      console.error("Failed to parse memory extraction:", error);
      return [];
    }

    const memories: Memory[] = [];

    // Store memories
    for (const memoryData of extractedData.memories || []) {
      try {
        const memory = await this.storeMemory({
          type: memoryData.type,
          category: memoryData.category,
          content: memoryData.content,
          context: memoryData.context,
          confidence: memoryData.confidence,
          importance: memoryData.importance,
          emotionalWeight: memoryData.emotional_weight || 0.0,
          tags: memoryData.tags || [],
          expiresAt: memoryData.expires_at
            ? new Date(memoryData.expires_at)
            : null,
          sourceConversationId: conversationId,
        });
        memories.push(memory);
      } catch (error) {
        console.error("Failed to store memory:", memoryData, error);
      }
    }

    // Store memory associations
    for (const association of extractedData.associations || []) {
      try {
        await this.storeMemoryAssociation(association);
      } catch (error) {
        console.error(
          "Failed to store memory association:",
          association,
          error
        );
      }
    }

    return memories;
  }

  private async storeMemory(memoryData: {
    type: string;
    category?: string;
    content: string;
    context: string;
    confidence: number;
    importance: number;
    emotionalWeight: number;
    tags: string[];
    expiresAt?: Date | null;
    sourceConversationId: string;
  }): Promise<Memory> {
    const client = await this.postgres.connect();

    try {
      await client.query("BEGIN");

      // Store in PostgreSQL
      const result = await client.query(
        `
        INSERT INTO beep_boop.user_memories (
          user_id, memory_type, category, content, context, confidence_score, 
          importance_score, emotional_weight, source_conversation_id, tags, expires_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING *
      `,
        [
          this.userId,
          memoryData.type,
          memoryData.category || "general",
          memoryData.content,
          memoryData.context,
          memoryData.confidence,
          memoryData.importance,
          memoryData.emotionalWeight,
          memoryData.sourceConversationId,
          memoryData.tags,
          memoryData.expiresAt,
        ]
      );

      const memory = result.rows[0];

      // Store embedding in ChromaDB
      const collectionName = `beep_boop_memories_${this.userId}`;
      const collection = await this.chroma.getCollection({
        name: collectionName,
      });

      await collection.add({
        ids: [memory.id],
        documents: [memoryData.content],
        metadatas: [
          {
            type: memoryData.type,
            category: memoryData.category || "general",
            context: memoryData.context,
            confidence: memoryData.confidence,
            importance: memoryData.importance,
            emotional_weight: memoryData.emotionalWeight,
            tags: memoryData.tags.join(","),
            source_conversation_id: memoryData.sourceConversationId,
            created_at: memory.created_at.toISOString(),
            expires_at: memoryData.expiresAt?.toISOString() || null,
          },
        ],
      });

      // Update embedding_id in PostgreSQL
      await client.query(
        "UPDATE beep_boop.user_memories SET embedding_id = $1 WHERE id = $2",
        [memory.id, memory.id]
      );

      // Store relationships in Neo4j
      await this.storeMemoryRelationships(memory, memoryData.tags);

      await client.query("COMMIT");

      return this.mapRowToMemory(memory);
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  // Additional implementation methods...
  // [Continue with remaining methods for memory management, Neo4j operations, etc.]

  private mapRowToMemory(row: any): Memory {
    return {
      id: row.id,
      userId: row.user_id,
      type: row.memory_type,
      category: row.category,
      content: row.content,
      context: row.context,
      confidence: row.confidence_score,
      importance: row.importance_score,
      emotionalWeight: row.emotional_weight || 0,
      tags: row.tags || [],
      embeddingId: row.embedding_id,
      verified: row.verified || false,
      createdAt: row.created_at,
      lastReferenced: row.last_referenced,
      referenceCount: row.reference_count || 0,
      expiresAt: row.expires_at,
    };
  }

  async cleanup(): Promise<void> {
    await this.postgres.end();
    await this.neo4j.close();
  }
}
```

I'll continue with the rest of the Beep-Boop recipe including the remaining implementation details, types, examples, and then move on to the other recipes. Would you like me to continue with the complete Beep-Boop implementation and then create the other recipe files?
