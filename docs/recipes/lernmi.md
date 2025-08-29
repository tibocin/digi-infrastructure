# Recipe 3: LERNMI - Reinforcement Learning Agent

**Overview**: An Ollama-powered reinforcement AI agent that interacts with Beep-Boop for continuous training and improvement through feedback loops, performance optimization, and adaptive learning strategies.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Learning Framework](#learning-framework)
- [Project Setup](#project-setup)
- [Database Schema Design](#database-schema-design)
- [Core Implementation](#core-implementation)
- [Training Episodes](#training-episodes)
- [Reward System](#reward-system)
- [Prompt Scaffolding](#prompt-scaffolding)
- [Learning Algorithms](#learning-algorithms)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Monitoring & Analytics](#monitoring--analytics)

## Architecture Overview

LERNMI is a reinforcement learning system that improves its performance through interactions with Beep-Boop (or other AI agents). It uses feedback loops, reward signals, and adaptive learning to continuously optimize its responses and decision-making capabilities.

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Training     │  │     LERNMI      │  │   Beep-Boop     │
│   Environment   │──┤   RL Agent      │──┤   (Teacher)     │
│                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │                      │
                     ┌─────────────────┐    ┌─────────────────┐
                     │   Experience    │    │   Performance   │
                     │   Memory &      │    │   Metrics &     │
                     │   Replay        │    │   Feedback      │
                     └─────────────────┘    └─────────────────┘
                              │
                     ┌─────────────────┐
                     │   PCS Service   │
                     │ (Dynamic        │
                     │  Prompts)       │
                     └─────────────────┘
```

### Key Components

1. **Experience Replay**: Store and learn from past interactions
2. **Reward System**: Multi-dimensional feedback mechanism
3. **Policy Network**: Decision-making strategy optimization
4. **Environment Simulation**: Controlled training scenarios
5. **Performance Analytics**: Continuous improvement tracking

## Learning Framework

### Learning Paradigms

#### 1. **Imitation Learning**

- Learn from Beep-Boop's successful interactions
- Copy high-performance response patterns
- Adapt communication styles and problem-solving approaches

#### 2. **Reinforcement Learning**

- Receive rewards for successful interactions
- Learn optimal policies through trial and error
- Explore new strategies while exploiting known good ones

#### 3. **Meta-Learning**

- Learn how to learn more efficiently
- Adapt quickly to new tasks and domains
- Transfer knowledge across different interaction contexts

#### 4. **Active Learning**

- Identify areas where more training is needed
- Request specific feedback on uncertain decisions
- Optimize data collection for maximum learning impact

## Project Setup

### Environment Configuration

```bash
# Create project directory
mkdir lernmi-rl-agent && cd lernmi-rl-agent

# Initialize Node.js project
npm init -y

# Install core dependencies
npm install @pcs/typescript-sdk pg qdrant-client redis
npm install @types/node typescript ts-node
npm install dotenv uuid mathjs

# Install ML and RL dependencies
npm install tensorflow @tensorflow/tfjs-node
npm install ml-matrix reinforcement-learning
npm install d3 plotly.js # for visualization

# Install reinforcement learning libraries
npm install gym-js reward-system
npm install experience-replay policy-gradient

# Create project structure
mkdir -p src/{agents,environments,policies,rewards}
mkdir -p src/{training,evaluation,visualization}
mkdir -p config tests examples data
```

### Environment Variables (.env)

```bash
# PCS Configuration
PCS_BASE_URL=http://localhost:8000
PCS_API_KEY=your_lernmi_api_key
PCS_APP_ID=lernmi-v1

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=digi
POSTGRES_USER=digi
POSTGRES_PASSWORD=your_postgres_password

# Redis Configuration (for experience replay)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Qdrant Configuration (for memory storage)
QDRANT_URL=http://localhost:6333

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:7b
OLLAMA_REASONING_MODEL=llama3.1:13b

# Beep-Boop Integration
BEEP_BOOP_API_URL=http://localhost:8001
BEEP_BOOP_API_KEY=your_beep_boop_key

# Learning Configuration
LEARNING_RATE=0.001
DISCOUNT_FACTOR=0.95
EXPLORATION_RATE=0.1
EXPLORATION_DECAY=0.995
BATCH_SIZE=32
MEMORY_CAPACITY=10000
UPDATE_FREQUENCY=100

# Training Configuration
EPISODES_PER_TRAINING=1000
MAX_STEPS_PER_EPISODE=100
EVALUATION_FREQUENCY=50
CHECKPOINT_FREQUENCY=200

# Monitoring Configuration
TENSORBOARD_PORT=6006
METRICS_PORT=8080
LOG_LEVEL=debug
```

## Database Schema Design

### PostgreSQL Schema

```sql
-- Create LERNMI schema
CREATE SCHEMA IF NOT EXISTS lernmi;

-- Learning episodes and sessions
CREATE TABLE lernmi.episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    episode_number INTEGER NOT NULL,
    environment_type VARCHAR(100), -- conversation, problem_solving, creative, analytical
    scenario_description TEXT,
    initial_state JSONB NOT NULL,
    final_state JSONB NOT NULL,
    total_reward FLOAT NOT NULL,
    step_count INTEGER NOT NULL,
    duration_seconds INTEGER,
    success BOOLEAN,
    completion_reason VARCHAR(100), -- completed, timeout, failure, early_termination
    learning_metrics JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Individual steps within episodes
CREATE TABLE lernmi.episode_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES lernmi.episodes(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    state_before JSONB NOT NULL,
    action_taken JSONB NOT NULL,
    reward_received FLOAT NOT NULL,
    state_after JSONB NOT NULL,
    action_probabilities JSONB, -- probability distribution over possible actions
    value_estimate FLOAT, -- critic's value estimate
    advantage FLOAT, -- advantage estimate for policy gradient
    teacher_feedback TEXT, -- feedback from Beep-Boop or other teacher
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Experience replay buffer
CREATE TABLE lernmi.experience_buffer (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward FLOAT NOT NULL,
    next_state JSONB NOT NULL,
    done BOOLEAN DEFAULT FALSE,
    priority FLOAT DEFAULT 1.0, -- for prioritized experience replay
    episode_id UUID REFERENCES lernmi.episodes(id),
    step_id UUID REFERENCES lernmi.episode_steps(id),
    learning_context VARCHAR(100), -- type of interaction this came from
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_sampled_at TIMESTAMP WITH TIME ZONE
);

-- Model checkpoints and versions
CREATE TABLE lernmi.model_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    checkpoint_type VARCHAR(50), -- periodic, best_performance, before_update, manual
    model_architecture JSONB NOT NULL,
    model_weights BYTEA, -- serialized model weights
    hyperparameters JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    training_episodes INTEGER NOT NULL,
    training_steps INTEGER NOT NULL,
    validation_score FLOAT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning performance metrics
CREATE TABLE lernmi.performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES lernmi.episodes(id),
    checkpoint_id UUID REFERENCES lernmi.model_checkpoints(id),
    metric_type VARCHAR(100), -- reward, accuracy, loss, convergence, exploration
    metric_name VARCHAR(255),
    metric_value FLOAT,
    metric_context JSONB DEFAULT '{}',
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Reward configurations and adaptations
CREATE TABLE lernmi.reward_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    reward_function JSONB NOT NULL, -- configuration for reward calculation
    weight_factors JSONB NOT NULL, -- weights for different reward components
    active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    effectiveness_score FLOAT, -- how well this reward config works
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Teacher interactions and feedback
CREATE TABLE lernmi.teacher_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES lernmi.episodes(id),
    step_id UUID REFERENCES lernmi.episode_steps(id),
    teacher_id VARCHAR(255), -- beep_boop, human, other_agent
    interaction_type VARCHAR(100), -- feedback, correction, guidance, demonstration
    teacher_input TEXT,
    lernmi_response TEXT,
    teacher_rating INTEGER CHECK (teacher_rating >= 1 AND teacher_rating <= 5),
    feedback_content TEXT,
    improvement_suggestions JSONB DEFAULT '[]',
    learning_objectives JSONB DEFAULT '[]',
    context JSONB DEFAULT '{}',
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning objectives and curriculum
CREATE TABLE lernmi.learning_objectives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    objective_type VARCHAR(100), -- skill, knowledge, behavior, strategy
    difficulty_level INTEGER CHECK (difficulty_level >= 1 AND difficulty_level <= 10),
    prerequisites UUID[], -- array of other objective IDs
    success_criteria JSONB NOT NULL,
    current_progress FLOAT DEFAULT 0.0 CHECK (current_progress >= 0.0 AND current_progress <= 1.0),
    target_performance FLOAT,
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 5),
    active BOOLEAN DEFAULT TRUE,
    achieved BOOLEAN DEFAULT FALSE,
    achieved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training schedules and curriculum progression
CREATE TABLE lernmi.training_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    schedule_type VARCHAR(50), -- fixed, adaptive, curriculum, progressive
    parameters JSONB NOT NULL,
    current_phase VARCHAR(100),
    phase_progress FLOAT DEFAULT 0.0,
    total_phases INTEGER,
    active BOOLEAN DEFAULT TRUE,
    effectiveness_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Knowledge transfer and generalization tracking
CREATE TABLE lernmi.knowledge_transfer (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_domain VARCHAR(255),
    target_domain VARCHAR(255),
    transfer_type VARCHAR(100), -- skill, pattern, strategy, knowledge
    transfer_success FLOAT CHECK (transfer_success >= 0.0 AND transfer_success <= 1.0),
    evidence JSONB DEFAULT '{}',
    episodes_to_transfer INTEGER, -- how many episodes it took to transfer
    performance_before FLOAT,
    performance_after FLOAT,
    confidence FLOAT DEFAULT 0.5,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_episodes_session_id ON lernmi.episodes(session_id);
CREATE INDEX idx_episodes_environment_type ON lernmi.episodes(environment_type);
CREATE INDEX idx_episodes_total_reward ON lernmi.episodes(total_reward DESC);
CREATE INDEX idx_episodes_started_at ON lernmi.episodes(started_at DESC);
CREATE INDEX idx_episode_steps_episode_id ON lernmi.episode_steps(episode_id);
CREATE INDEX idx_episode_steps_step_number ON lernmi.episode_steps(step_number);
CREATE INDEX idx_episode_steps_reward ON lernmi.episode_steps(reward_received DESC);
CREATE INDEX idx_experience_buffer_priority ON lernmi.experience_buffer(priority DESC);
CREATE INDEX idx_experience_buffer_created_at ON lernmi.experience_buffer(created_at DESC);
CREATE INDEX idx_experience_buffer_context ON lernmi.experience_buffer(learning_context);
CREATE INDEX idx_model_checkpoints_version ON lernmi.model_checkpoints(model_name, version DESC);
CREATE INDEX idx_performance_metrics_type ON lernmi.performance_metrics(metric_type);
CREATE INDEX idx_performance_metrics_measured_at ON lernmi.performance_metrics(measured_at DESC);
CREATE INDEX idx_teacher_interactions_episode_id ON lernmi.teacher_interactions(episode_id);
CREATE INDEX idx_teacher_interactions_teacher_id ON lernmi.teacher_interactions(teacher_id);
CREATE INDEX idx_learning_objectives_progress ON lernmi.learning_objectives(current_progress DESC);
CREATE INDEX idx_knowledge_transfer_domains ON lernmi.knowledge_transfer(source_domain, target_domain);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION lernmi.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_reward_configurations_updated_at
    BEFORE UPDATE ON lernmi.reward_configurations
    FOR EACH ROW EXECUTE FUNCTION lernmi.update_updated_at_column();

CREATE TRIGGER update_learning_objectives_updated_at
    BEFORE UPDATE ON lernmi.learning_objectives
    FOR EACH ROW EXECUTE FUNCTION lernmi.update_updated_at_column();

CREATE TRIGGER update_training_schedules_updated_at
    BEFORE UPDATE ON lernmi.training_schedules
    FOR EACH ROW EXECUTE FUNCTION lernmi.update_updated_at_column();

-- Function to update experience buffer sampling
CREATE OR REPLACE FUNCTION lernmi.update_experience_sampled()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE lernmi.experience_buffer
    SET last_sampled_at = NOW()
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ language 'plpgsql';
```

## Core Implementation

### Main LERNMI Agent Class

```typescript
// src/agents/lernmi-agent.ts
import { PCSClient } from "@pcs/typescript-sdk";
import { Pool, PoolClient } from "pg";
import { createClient } from "redis";
import * as tf from "@tensorflow/tfjs-node";
import { v4 as uuidv4 } from "uuid";
import {
  Episode,
  EpisodeStep,
  ExperienceBuffer,
  RewardSignal,
  LearningObjective,
  PerformanceMetrics,
} from "../types/lernmi-types";

export class LernmiAgent {
  private pcs: PCSClient;
  private postgres: Pool;
  private redis: any;
  private beepBoopClient: any;

  // Learning components
  private policyNetwork: tf.LayersModel;
  private valueNetwork: tf.LayersModel;
  private experienceBuffer: ExperienceBuffer;
  private rewardSystem: RewardSystem;

  // Training state
  private currentEpisode?: Episode;
  private learningRate: number;
  private explorationRate: number;
  private isTraining: boolean = false;

  // Performance tracking
  private performanceHistory: PerformanceMetrics[] = [];
  private learningObjectives: LearningObjective[] = [];

  constructor(config: {
    pcsConfig: any;
    postgresConfig: any;
    redisConfig: any;
    beepBoopConfig: any;
    learningConfig: {
      learningRate: number;
      explorationRate: number;
      discountFactor: number;
      batchSize: number;
      memoryCapacity: number;
    };
  }) {
    this.pcs = new PCSClient(config.pcsConfig);
    this.postgres = new Pool(config.postgresConfig);
    this.redis = createClient(config.redisConfig);

    this.learningRate = config.learningConfig.learningRate;
    this.explorationRate = config.learningConfig.explorationRate;

    this.initializeNeuralNetworks();
    this.initializeExperienceBuffer(config.learningConfig);
    this.initializeRewardSystem();
  }

  private initializeNeuralNetworks(): void {
    // Policy Network (Actor)
    this.policyNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [128], // state representation size
          units: 256,
          activation: "relu",
          name: "policy_hidden_1",
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({
          units: 128,
          activation: "relu",
          name: "policy_hidden_2",
        }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({
          units: 64, // action space size
          activation: "softmax",
          name: "policy_output",
        }),
      ],
    });

    // Value Network (Critic)
    this.valueNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [128],
          units: 256,
          activation: "relu",
          name: "value_hidden_1",
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({
          units: 128,
          activation: "relu",
          name: "value_hidden_2",
        }),
        tf.layers.dense({
          units: 1,
          activation: "linear",
          name: "value_output",
        }),
      ],
    });

    // Compile networks with optimizers
    this.policyNetwork.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    this.valueNetwork.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError",
      metrics: ["mae"],
    });
  }

  async startEpisode(scenario: {
    type: string;
    description: string;
    initialContext: any;
    learningObjectives: string[];
  }): Promise<string> {
    const episodeId = uuidv4();
    const sessionId = uuidv4();

    this.currentEpisode = {
      id: episodeId,
      sessionId,
      episodeNumber: await this.getNextEpisodeNumber(),
      environmentType: scenario.type,
      scenarioDescription: scenario.description,
      initialState: scenario.initialContext,
      totalReward: 0,
      stepCount: 0,
      steps: [],
      startedAt: new Date(),
      status: "active",
    };

    // Store episode in database
    await this.storeEpisode(this.currentEpisode);

    console.log(`Started episode ${episodeId} - ${scenario.description}`);
    return episodeId;
  }

  async takeAction(state: any): Promise<{
    action: any;
    confidence: number;
    reasoning: string;
  }> {
    if (!this.currentEpisode) {
      throw new Error("No active episode");
    }

    // Convert state to tensor
    const stateTensor = this.stateToTensor(state);

    // Get action probabilities from policy network
    const actionProbs = this.policyNetwork.predict(stateTensor) as tf.Tensor;
    const actionProbsArray = await actionProbs.data();

    // Choose action (epsilon-greedy exploration)
    let actionIndex: number;
    if (Math.random() < this.explorationRate) {
      // Explore: random action
      actionIndex = Math.floor(Math.random() * actionProbsArray.length);
    } else {
      // Exploit: best action according to policy
      actionIndex = actionProbsArray.indexOf(Math.max(...actionProbsArray));
    }

    // Generate action using PCS
    const action = await this.generateActionFromIndex(actionIndex, state);

    // Get value estimate
    const valueEstimate = this.valueNetwork.predict(stateTensor) as tf.Tensor;
    const confidence = (await valueEstimate.data())[0];

    // Clean up tensors
    stateTensor.dispose();
    actionProbs.dispose();
    valueEstimate.dispose();

    return {
      action,
      confidence,
      reasoning: `Selected action ${actionIndex} with confidence ${confidence.toFixed(
        3
      )}`,
    };
  }

  private async generateActionFromIndex(
    actionIndex: number,
    state: any
  ): Promise<any> {
    // Map action index to actual actions using PCS
    const actionMapping = await this.pcs.generatePrompt(
      "lernmi_action_generation",
      {
        context: {
          action_index: actionIndex.toString(),
          current_state: JSON.stringify(state),
          available_actions: JSON.stringify(this.getAvailableActions()),
          episode_context: JSON.stringify(
            this.currentEpisode?.scenarioDescription
          ),
        },
      }
    );

    try {
      return JSON.parse(actionMapping.generated_prompt);
    } catch (error) {
      console.error("Failed to parse action mapping:", error);
      return { type: "default", content: "I need to think about this more." };
    }
  }

  async receiveReward(
    previousState: any,
    action: any,
    newState: any,
    teacherFeedback?: {
      rating: number;
      feedback: string;
      suggestions: string[];
    }
  ): Promise<number> {
    if (!this.currentEpisode) {
      throw new Error("No active episode");
    }

    // Calculate multi-dimensional reward
    const rewardSignal = await this.rewardSystem.calculateReward({
      previousState,
      action,
      newState,
      teacherFeedback,
      episodeContext: this.currentEpisode,
    });

    // Create episode step
    const step: EpisodeStep = {
      id: uuidv4(),
      episodeId: this.currentEpisode.id,
      stepNumber: this.currentEpisode.stepCount,
      stateBefore: previousState,
      actionTaken: action,
      rewardReceived: rewardSignal.totalReward,
      stateAfter: newState,
      teacherFeedback: teacherFeedback?.feedback,
      timestamp: new Date(),
      metadata: {
        rewardComponents: rewardSignal.components,
        confidence: rewardSignal.confidence,
      },
    };

    // Add to current episode
    this.currentEpisode.steps.push(step);
    this.currentEpisode.stepCount++;
    this.currentEpisode.totalReward += rewardSignal.totalReward;

    // Store step in database
    await this.storeEpisodeStep(step);

    // Add to experience buffer for later training
    await this.experienceBuffer.add({
      state: previousState,
      action,
      reward: rewardSignal.totalReward,
      nextState: newState,
      done: false, // Will be updated when episode ends
      episodeId: this.currentEpisode.id,
      stepId: step.id,
    });

    // Store teacher interaction if provided
    if (teacherFeedback) {
      await this.storeTeacherInteraction({
        episodeId: this.currentEpisode.id,
        stepId: step.id,
        teacherId: "beep_boop", // or other teacher identifier
        interactionType: "feedback",
        teacherInput: teacherFeedback.feedback,
        lernmiResponse: JSON.stringify(action),
        teacherRating: teacherFeedback.rating,
        feedbackContent: teacherFeedback.feedback,
        improvementSuggestions: teacherFeedback.suggestions,
      });
    }

    console.log(
      `Step ${step.stepNumber}: Reward = ${rewardSignal.totalReward.toFixed(3)}`
    );
    return rewardSignal.totalReward;
  }

  async endEpisode(reason: string = "completed"): Promise<{
    episodeId: string;
    totalReward: number;
    stepCount: number;
    learningMetrics: any;
  }> {
    if (!this.currentEpisode) {
      throw new Error("No active episode");
    }

    this.currentEpisode.completedAt = new Date();
    this.currentEpisode.completionReason = reason;
    this.currentEpisode.status = "completed";

    // Mark last experience as terminal
    if (this.currentEpisode.steps.length > 0) {
      const lastStep =
        this.currentEpisode.steps[this.currentEpisode.steps.length - 1];
      await this.experienceBuffer.markAsTerminal(lastStep.id);
    }

    // Calculate learning metrics
    const learningMetrics = await this.calculateEpisodeLearningMetrics(
      this.currentEpisode
    );
    this.currentEpisode.learningMetrics = learningMetrics;

    // Update episode in database
    await this.updateEpisode(this.currentEpisode);

    // Trigger learning if we have enough experience
    if ((await this.experienceBuffer.size()) >= 32) {
      // batch size
      await this.performLearningUpdate();
    }

    // Update learning objectives progress
    await this.updateLearningObjectivesProgress(this.currentEpisode);

    const result = {
      episodeId: this.currentEpisode.id,
      totalReward: this.currentEpisode.totalReward,
      stepCount: this.currentEpisode.stepCount,
      learningMetrics,
    };

    console.log(
      `Episode ${this.currentEpisode.id} completed: ${JSON.stringify(result)}`
    );
    this.currentEpisode = undefined;

    return result;
  }

  private async performLearningUpdate(): Promise<void> {
    console.log("Performing learning update...");

    // Sample batch from experience buffer
    const batch = await this.experienceBuffer.sample(32);

    if (batch.length === 0) return;

    // Prepare training data
    const states = batch.map((exp) => exp.state);
    const actions = batch.map((exp) => exp.action);
    const rewards = batch.map((exp) => exp.reward);
    const nextStates = batch.map((exp) => exp.nextState);
    const dones = batch.map((exp) => exp.done);

    // Convert to tensors
    const stateTensors = tf.tensor2d(states.map((s) => this.stateToVector(s)));
    const nextStateTensors = tf.tensor2d(
      nextStates.map((s) => this.stateToVector(s))
    );
    const rewardTensors = tf.tensor1d(rewards);

    // Calculate value targets using Bellman equation
    const currentValues = this.valueNetwork.predict(stateTensors) as tf.Tensor;
    const nextValues = this.valueNetwork.predict(nextStateTensors) as tf.Tensor;

    // Value targets: r + γ * V(s') * (1 - done)
    const valueTargets = tf.add(
      rewardTensors,
      tf.mul(
        tf.mul(nextValues.squeeze(), tf.scalar(0.95)), // discount factor
        tf.sub(tf.scalar(1), tf.tensor1d(dones.map((d) => (d ? 1 : 0))))
      )
    );

    // Calculate advantages for policy update
    const advantages = tf.sub(valueTargets, currentValues.squeeze());

    // Update value network
    await this.valueNetwork.fit(stateTensors, valueTargets, {
      epochs: 1,
      batchSize: 32,
      verbose: 0,
    });

    // Update policy network (simplified policy gradient)
    const actionTensors = tf.tensor2d(
      actions.map((a) => this.actionToVector(a))
    );

    // Custom training step for policy network
    await this.updatePolicyNetwork(stateTensors, actionTensors, advantages);

    // Clean up tensors
    stateTensors.dispose();
    nextStateTensors.dispose();
    rewardTensors.dispose();
    currentValues.dispose();
    nextValues.dispose();
    valueTargets.dispose();
    advantages.dispose();
    actionTensors.dispose();

    // Decay exploration rate
    this.explorationRate = Math.max(0.01, this.explorationRate * 0.995);

    console.log("Learning update completed");
  }

  private async updatePolicyNetwork(
    states: tf.Tensor,
    actions: tf.Tensor,
    advantages: tf.Tensor
  ): Promise<void> {
    const optimizer = tf.train.adam(this.learningRate);

    const loss = tf.tidy(() => {
      const predictions = this.policyNetwork.predict(states) as tf.Tensor;
      const actionProbs = tf.sum(tf.mul(predictions, actions), 1);
      const logProbs = tf.log(tf.clipByValue(actionProbs, 1e-8, 1.0));
      return tf.mean(tf.mul(tf.neg(logProbs), advantages));
    });

    const grads = tf.variableGrads(loss, this.policyNetwork.trainableWeights);
    optimizer.applyGradients(grads.grads);

    loss.dispose();
    Object.values(grads.grads).forEach((grad) => grad.dispose());
  }

  async interactWithBeepBoop(
    message: string,
    context?: any
  ): Promise<{
    response: string;
    feedback: any;
    learningSignals: any;
  }> {
    // Send message to Beep-Boop and receive structured response
    const beepBoopResponse = await this.beepBoopClient.sendMessage({
      message,
      context: {
        ...context,
        source: "lernmi",
        training_mode: true,
        request_feedback: true,
      },
    });

    // Extract learning signals from Beep-Boop's response
    const learningSignals = await this.extractLearningSignals(beepBoopResponse);

    // Store interaction for future analysis
    await this.storeBeepBoopInteraction({
      lernmiMessage: message,
      beepBoopResponse: beepBoopResponse.response,
      context,
      learningSignals,
      timestamp: new Date(),
    });

    return {
      response: beepBoopResponse.response,
      feedback: beepBoopResponse.feedback,
      learningSignals,
    };
  }

  // State and action encoding methods
  private stateToTensor(state: any): tf.Tensor {
    const vector = this.stateToVector(state);
    return tf.tensor2d([vector]);
  }

  private stateToVector(state: any): number[] {
    // Convert complex state to fixed-size numerical vector
    // This is a simplified implementation - in practice, you'd want more sophisticated encoding
    const vector = new Array(128).fill(0);

    // Encode different aspects of the state
    if (state.context) {
      // Encode context features
      vector[0] = state.context.length || 0;
      vector[1] = state.context.includes("question") ? 1 : 0;
      vector[2] = state.context.includes("problem") ? 1 : 0;
      // ... more context features
    }

    if (state.conversation_history) {
      vector[10] = Math.min(state.conversation_history.length / 10, 1.0);
      // ... encode conversation features
    }

    if (state.user_mood) {
      const moodMap = {
        happy: 1,
        sad: -1,
        neutral: 0,
        excited: 0.8,
        frustrated: -0.8,
      };
      vector[20] = moodMap[state.user_mood] || 0;
    }

    // Add temporal features
    const hour = new Date().getHours();
    vector[30] = Math.sin((2 * Math.PI * hour) / 24); // cyclical time encoding
    vector[31] = Math.cos((2 * Math.PI * hour) / 24);

    return vector;
  }

  private actionToVector(action: any): number[] {
    // Convert action to one-hot or multi-hot vector
    const vector = new Array(64).fill(0);

    if (action.type) {
      const actionTypes = [
        "question",
        "answer",
        "clarification",
        "suggestion",
        "analysis",
      ];
      const typeIndex = actionTypes.indexOf(action.type);
      if (typeIndex >= 0) vector[typeIndex] = 1;
    }

    if (action.confidence) {
      vector[10] = action.confidence;
    }

    if (action.length) {
      vector[11] = Math.min(action.length / 1000, 1.0); // normalized length
    }

    return vector;
  }

  // Utility methods for getting available actions
  private getAvailableActions(): string[] {
    return [
      "ask_clarifying_question",
      "provide_direct_answer",
      "suggest_alternative",
      "request_more_context",
      "analyze_problem",
      "break_down_task",
      "summarize_understanding",
      "admit_uncertainty",
    ];
  }

  // Database operations
  private async storeEpisode(episode: Episode): Promise<void> {
    const client = await this.postgres.connect();
    try {
      await client.query(
        `
        INSERT INTO lernmi.episodes (
          id, session_id, episode_number, environment_type, scenario_description,
          initial_state, total_reward, step_count, started_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      `,
        [
          episode.id,
          episode.sessionId,
          episode.episodeNumber,
          episode.environmentType,
          episode.scenarioDescription,
          JSON.stringify(episode.initialState),
          episode.totalReward,
          episode.stepCount,
          episode.startedAt,
        ]
      );
    } finally {
      client.release();
    }
  }

  private async storeEpisodeStep(step: EpisodeStep): Promise<void> {
    const client = await this.postgres.connect();
    try {
      await client.query(
        `
        INSERT INTO lernmi.episode_steps (
          id, episode_id, step_number, state_before, action_taken, reward_received,
          state_after, teacher_feedback, metadata, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      `,
        [
          step.id,
          step.episodeId,
          step.stepNumber,
          JSON.stringify(step.stateBefore),
          JSON.stringify(step.actionTaken),
          step.rewardReceived,
          JSON.stringify(step.stateAfter),
          step.teacherFeedback,
          JSON.stringify(step.metadata),
          step.timestamp,
        ]
      );
    } finally {
      client.release();
    }
  }

  // Additional methods for training, evaluation, checkpointing, etc.

  async saveCheckpoint(name: string, notes?: string): Promise<string> {
    const checkpointId = uuidv4();

    // Save model weights
    const policyWeights = await this.serializeModel(this.policyNetwork);
    const valueWeights = await this.serializeModel(this.valueNetwork);

    const client = await this.postgres.connect();
    try {
      await client.query(
        `
        INSERT INTO lernmi.model_checkpoints (
          id, model_name, version, checkpoint_type, model_architecture,
          hyperparameters, performance_metrics, training_episodes, notes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      `,
        [
          checkpointId,
          "lernmi_agent",
          await this.getNextVersion(),
          "manual",
          JSON.stringify({
            policy_architecture: this.policyNetwork.toJSON(),
            value_architecture: this.valueNetwork.toJSON(),
            policy_weights: policyWeights,
            value_weights: valueWeights,
          }),
          JSON.stringify({
            learning_rate: this.learningRate,
            exploration_rate: this.explorationRate,
            discount_factor: 0.95,
          }),
          JSON.stringify(await this.getCurrentPerformanceMetrics()),
          await this.getTotalEpisodes(),
          notes,
        ]
      );
    } finally {
      client.release();
    }

    return checkpointId;
  }

  async cleanup(): Promise<void> {
    await this.postgres.end();
    await this.redis.quit();
  }
}
```

I'll continue with the remaining components of LERNMI including the reward system, training episodes, and usage examples. Would you like me to continue with the complete implementation and then move on to the final Bitscrow recipe?
