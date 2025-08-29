# Recipe 2: DEVAO - Virtual Development Shop

**Overview**: A multi-agent development team using LangGraph to coordinate 10 specialized Ollama agents that autonomously build, test, monitor, and maintain software projects within the Digi Infrastructure.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent Specializations](#agent-specializations)
- [Project Setup](#project-setup)
- [Database Schema Design](#database-schema-design)
- [LangGraph Workflow](#langgraph-workflow)
- [Agent Implementations](#agent-implementations)
- [Prompt Scaffolding](#prompt-scaffolding)
- [Coordination System](#coordination-system)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Monitoring & Analytics](#monitoring--analytics)

## Architecture Overview

DEVAO operates as a virtual software development company with specialized AI agents working together to deliver complete software projects. Each agent has distinct capabilities and communicates through a centralized coordination system powered by LangGraph.

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Project       │  │   LangGraph     │  │   Agent Pool    │
│   Requirements  │──┤   Orchestrator  │──┤   (10 Agents)   │
│                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │                      │
                     ┌─────────────────┐    ┌─────────────────┐
                     │   PCS Service   │    │   Git & CI/CD   │
                     │ (Coordination   │    │   Integration   │
                     │  & Memory)      │    │                 │
                     └─────────────────┘    └─────────────────┘
                              │
                     ┌─────────────────┐
                     │   Monitoring    │
                     │   & Analytics   │
                     │   Dashboard     │
                     └─────────────────┘
```

### Key Features

1. **Autonomous Development**: Agents work independently and collaboratively
2. **Complete SDLC**: From requirements to deployment and maintenance
3. **Quality Assurance**: Built-in testing, code review, and security checks
4. **Continuous Integration**: Automated build, test, and deployment pipelines
5. **Performance Monitoring**: Real-time performance tracking and optimization

## Agent Specializations

The DEVAO system consists of 10 specialized agents:

### 1. **Architect Agent** 🏗️

- **Role**: System design and architecture decisions
- **Responsibilities**:
  - Analyze requirements and design system architecture
  - Choose technology stacks and design patterns
  - Define component interfaces and data flows
  - Create technical specifications and documentation

### 2. **Frontend Agent** 🎨

- **Role**: UI/UX development and client-side implementation
- **Responsibilities**:
  - Design and implement user interfaces
  - Handle responsive design and accessibility
  - Implement client-side state management
  - Optimize frontend performance

### 3. **Backend Agent** ⚙️

- **Role**: Server-side logic and API development
- **Responsibilities**:
  - Implement business logic and APIs
  - Design and optimize database operations
  - Handle authentication and authorization
  - Implement caching and performance optimizations

### 4. **Database Agent** 🗄️

- **Role**: Data modeling and database operations
- **Responsibilities**:
  - Design database schemas and relationships
  - Optimize queries and indexing strategies
  - Handle data migrations and seeding
  - Implement backup and recovery procedures

### 5. **DevOps Agent** 🚀

- **Role**: Deployment, CI/CD, and infrastructure management
- **Responsibilities**:
  - Set up deployment pipelines and environments
  - Configure monitoring and logging systems
  - Manage infrastructure as code
  - Handle scaling and load balancing

### 6. **QA Agent** 🧪

- **Role**: Testing strategies and quality assurance
- **Responsibilities**:
  - Design and implement testing strategies
  - Create unit, integration, and e2e tests
  - Perform manual testing and bug reporting
  - Ensure code coverage and quality metrics

### 7. **Security Agent** 🔒

- **Role**: Security audits and vulnerability assessment
- **Responsibilities**:
  - Perform security audits and penetration testing
  - Implement security best practices
  - Monitor for vulnerabilities and threats
  - Ensure compliance with security standards

### 8. **Performance Agent** ⚡

- **Role**: Optimization and performance monitoring
- **Responsibilities**:
  - Monitor application performance metrics
  - Identify bottlenecks and optimization opportunities
  - Implement caching and performance improvements
  - Handle capacity planning and scaling

### 9. **Documentation Agent** 📚

- **Role**: Technical documentation and knowledge management
- **Responsibilities**:
  - Create and maintain technical documentation
  - Generate API documentation and user guides
  - Maintain code comments and inline documentation
  - Create tutorials and onboarding materials

### 10. **Project Manager Agent** 📋

- **Role**: Coordination, planning, and progress tracking
- **Responsibilities**:
  - Break down requirements into manageable tasks
  - Coordinate agent activities and dependencies
  - Track progress and manage timelines
  - Handle stakeholder communication

## Project Setup

### Environment Configuration

```bash
# Create project directory
mkdir devao-virtual-team && cd devao-virtual-team

# Initialize Node.js project
npm init -y

# Install core dependencies
npm install @pcs/typescript-sdk @langchain/langgraph
npm install pg qdrant-client neo4j-driver redis
npm install @types/node typescript ts-node
npm install dotenv uuid

# Install LangChain and AI dependencies
npm install @langchain/core @langchain/community
npm install @langchain/openai langchain
npm install ollama

# Install development and deployment tools
npm install simple-git dockerode
npm install @octokit/rest # for GitHub integration
npm install nodemailer # for notifications

# Create project structure
mkdir -p src/{agents,workflows,types,services,utils}
mkdir -p src/{prompts,coordinators,monitors}
mkdir -p config tests examples deployment
```

### Environment Variables (.env)

```bash
# PCS Configuration
PCS_BASE_URL=http://localhost:8000
PCS_API_KEY=your_devao_api_key
PCS_APP_ID=devao-v1

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=db_name
POSTGRES_USER=db_user
POSTGRES_PASSWORD=your_postgres_password

# Neo4j Configuration (for agent relationships)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Redis Configuration (for agent communication)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Qdrant Configuration (for knowledge base)
QDRANT_URL=http://localhost:6333

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:7b
OLLAMA_CODE_MODEL=codellama:7b

# Git Configuration
GIT_USERNAME=devao-bot
GIT_EMAIL=devao@yourcompany.com
GITHUB_TOKEN=your_github_token

# OpenAI Configuration (optional, for enhanced capabilities)
OPENAI_API_KEY=your_openai_key

# Docker Configuration
DOCKER_HOST=unix:///var/run/docker.sock

# CI/CD Configuration
JENKINS_URL=http://localhost:8080
JENKINS_USER=admin
JENKINS_TOKEN=your_jenkins_token

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3001

# Application Configuration
NODE_ENV=development
LOG_LEVEL=debug
MAX_CONCURRENT_AGENTS=5
TASK_TIMEOUT_MINUTES=30
PROJECT_WORKSPACE=/tmp/devao-projects
```

## Database Schema Design

### PostgreSQL Schema

```sql
-- Create DEVAO schema
CREATE SCHEMA IF NOT EXISTS devao;

-- Projects and their metadata
CREATE TABLE devao.projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    repository_url VARCHAR(500),
    repository_branch VARCHAR(100) DEFAULT 'main',
    status VARCHAR(50) DEFAULT 'planning', -- planning, development, testing, deployed, maintenance, completed
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 5), -- 1=low, 5=critical
    complexity_score INTEGER DEFAULT 1 CHECK (complexity_score >= 1 AND complexity_score <= 10),
    estimated_hours INTEGER,
    actual_hours INTEGER DEFAULT 0,
    target_completion TIMESTAMP WITH TIME ZONE,
    tech_stack JSONB DEFAULT '{}',
    requirements JSONB DEFAULT '{}',
    architecture JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Agent definitions and capabilities
CREATE TABLE devao.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    specialization VARCHAR(100) NOT NULL,
    model_config JSONB NOT NULL, -- Ollama model and parameters
    capabilities TEXT[],
    skills JSONB DEFAULT '{}', -- skill levels and expertise areas
    status VARCHAR(50) DEFAULT 'available', -- available, busy, offline, maintenance
    current_task_id UUID,
    max_concurrent_tasks INTEGER DEFAULT 1,
    performance_metrics JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}', -- coding style, tool preferences, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tasks and work items
CREATE TABLE devao.tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES devao.projects(id) ON DELETE CASCADE,
    parent_task_id UUID REFERENCES devao.tasks(id),
    assigned_agent_id UUID REFERENCES devao.agents(id),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    task_type VARCHAR(100) NOT NULL, -- feature, bug, test, deploy, refactor, documentation
    category VARCHAR(100), -- frontend, backend, database, devops, etc.
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 5),
    complexity INTEGER DEFAULT 1 CHECK (complexity >= 1 AND complexity <= 10),
    status VARCHAR(50) DEFAULT 'pending', -- pending, assigned, in_progress, review, completed, failed, blocked
    requirements JSONB DEFAULT '{}',
    acceptance_criteria JSONB DEFAULT '[]',
    deliverables JSONB DEFAULT '{}',
    estimated_hours INTEGER,
    actual_hours INTEGER DEFAULT 0,
    dependencies UUID[], -- Array of task IDs that must be completed first
    blockers JSONB DEFAULT '[]', -- Issues preventing progress
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    quality_score FLOAT, -- 0.0 to 1.0 based on reviews and tests
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent communications and decisions
CREATE TABLE devao.agent_communications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES devao.projects(id) ON DELETE CASCADE,
    task_id UUID REFERENCES devao.tasks(id),
    from_agent_id UUID REFERENCES devao.agents(id),
    to_agent_id UUID REFERENCES devao.agents(id),
    communication_type VARCHAR(50), -- decision, question, update, review, approval, notification
    subject VARCHAR(255),
    content TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    urgency VARCHAR(20) DEFAULT 'normal', -- low, normal, high, urgent
    response_required BOOLEAN DEFAULT FALSE,
    response_by TIMESTAMP WITH TIME ZONE,
    response_content TEXT,
    responded_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Code artifacts and deliverables
CREATE TABLE devao.artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES devao.projects(id) ON DELETE CASCADE,
    task_id UUID REFERENCES devao.tasks(id),
    git_repo VARCHAR(1000), -- Repository URL
    git_branch VARCHAR(255), -- Branch name
    git_commit_hash VARCHAR(40), -- Full commit SHA
    git_commit_message TEXT, -- Associated commit message
    git_commit_timestamp TIMESTAMP WITH TIME ZONE, -- When commit was made
    created_by_agent_id UUID REFERENCES devao.agents(id),
    artifact_type VARCHAR(100), -- code, test, documentation, config, schema, deployment
    file_path VARCHAR(1000),
    content TEXT,
    language VARCHAR(50), -- programming language or markup type
    framework VARCHAR(100), -- React, Express, etc.
    metadata JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    status VARCHAR(50) DEFAULT 'draft', -- draft, review, approved, deployed, deprecated
    quality_metrics JSONB DEFAULT '{}', -- complexity, coverage, performance metrics
    review_comments JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Project metrics and performance data
CREATE TABLE devao.project_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES devao.projects(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES devao.agents(id),
    metric_type VARCHAR(100), -- build_time, test_coverage, performance, bugs, code_quality
    metric_name VARCHAR(255),
    metric_value FLOAT,
    unit VARCHAR(50), -- seconds, percentage, count, etc.
    metadata JSONB DEFAULT '{}',
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow states and transitions
CREATE TABLE devao.workflow_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES devao.projects(id) ON DELETE CASCADE,
    current_phase VARCHAR(100), -- analysis, design, development, testing, deployment, maintenance
    state_data JSONB NOT NULL, -- Complete state information for LangGraph
    active_agents UUID[], -- Currently active agent IDs
    pending_decisions JSONB DEFAULT '[]',
    blockers JSONB DEFAULT '[]',
    next_actions JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent performance and learning
CREATE TABLE devao.agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES devao.agents(id) ON DELETE CASCADE,
    project_id UUID REFERENCES devao.projects(id),
    task_id UUID REFERENCES devao.tasks(id),
    performance_type VARCHAR(100), -- task_completion, code_quality, collaboration, problem_solving
    score FLOAT CHECK (score >= 0.0 AND score <= 1.0),
    details JSONB DEFAULT '{}',
    feedback TEXT,
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_projects_status ON devao.projects(status);
CREATE INDEX idx_projects_priority ON devao.projects(priority DESC);
CREATE INDEX idx_tasks_project_id ON devao.tasks(project_id);
CREATE INDEX idx_tasks_agent_id ON devao.tasks(assigned_agent_id);
CREATE INDEX idx_tasks_status ON devao.tasks(status);
CREATE INDEX idx_tasks_priority ON devao.tasks(priority DESC);
CREATE INDEX idx_tasks_dependencies ON devao.tasks USING GIN(dependencies);
CREATE INDEX idx_communications_project_id ON devao.agent_communications(project_id);
CREATE INDEX idx_communications_agents ON devao.agent_communications(from_agent_id, to_agent_id);
CREATE INDEX idx_artifacts_project_id ON devao.artifacts(project_id);
CREATE INDEX idx_artifacts_task_id ON devao.artifacts(task_id);
CREATE INDEX idx_artifacts_type ON devao.artifacts(artifact_type);
CREATE INDEX idx_metrics_project_id ON devao.project_metrics(project_id);
CREATE INDEX idx_metrics_type ON devao.project_metrics(metric_type);
CREATE INDEX idx_workflow_states_project_id ON devao.workflow_states(project_id);
CREATE INDEX idx_agent_performance_agent_id ON devao.agent_performance(agent_id);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION devao.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON devao.projects
    FOR EACH ROW EXECUTE FUNCTION devao.update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON devao.tasks
    FOR EACH ROW EXECUTE FUNCTION devao.update_updated_at_column();

CREATE TRIGGER update_artifacts_updated_at
    BEFORE UPDATE ON devao.artifacts
    FOR EACH ROW EXECUTE FUNCTION devao.update_updated_at_column();

CREATE TRIGGER update_workflow_states_updated_at
    BEFORE UPDATE ON devao.workflow_states
    FOR EACH ROW EXECUTE FUNCTION devao.update_updated_at_column();

-- Function to update agent last_active_at
CREATE OR REPLACE FUNCTION devao.update_agent_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE devao.agents
    SET last_active_at = NOW()
    WHERE id = NEW.from_agent_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agent_activity_on_communication
    AFTER INSERT ON devao.agent_communications
    FOR EACH ROW EXECUTE FUNCTION devao.update_agent_activity();
```

## LangGraph Workflow

### Workflow Definition

```typescript
// src/workflows/devao-workflow.ts
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { PCSClient } from "@pcs/typescript-sdk";
import { DevaoAgent } from "../agents/base-agent";
import {
  ArchitectAgent,
  FrontendAgent,
  BackendAgent,
  DatabaseAgent,
  DevOpsAgent,
  QAAgent,
  SecurityAgent,
  PerformanceAgent,
  DocumentationAgent,
  ProjectManagerAgent,
} from "../agents";

// Define the state structure for our workflow
const WorkflowState = Annotation.Root({
  project_id: Annotation<string>(),
  current_phase: Annotation<string>(),
  active_tasks: Annotation<any[]>(),
  agent_assignments: Annotation<Record<string, string>>(),
  project_context: Annotation<Record<string, any>>(),
  decisions: Annotation<any[]>(),
  artifacts: Annotation<any[]>(),
  blockers: Annotation<any[]>(),
  quality_metrics: Annotation<Record<string, number>>(),
  timeline: Annotation<Record<string, any>>(),
  next_actions: Annotation<string[]>(),
});

export class DevaoWorkflow {
  private pcs: PCSClient;
  private agents: Map<string, DevaoAgent>;
  private workflow: StateGraph<typeof WorkflowState.State>;

  constructor(config: {
    pcsConfig: any;
    postgresConfig: any;
    redisConfig: any;
    ollamaConfig: any;
  }) {
    this.pcs = new PCSClient(config.pcsConfig);
    this.agents = new Map();
    this.initializeAgents(config);
    this.buildWorkflow();
  }

  private initializeAgents(config: any): void {
    // Initialize all 10 specialized agents
    const agentConfigs = {
      architect: { ...config, specialization: "architecture" },
      frontend: { ...config, specialization: "frontend" },
      backend: { ...config, specialization: "backend" },
      database: { ...config, specialization: "database" },
      devops: { ...config, specialization: "devops" },
      qa: { ...config, specialization: "testing" },
      security: { ...config, specialization: "security" },
      performance: { ...config, specialization: "performance" },
      documentation: { ...config, specialization: "documentation" },
      project_manager: { ...config, specialization: "project_management" },
    };

    this.agents.set("architect", new ArchitectAgent(agentConfigs.architect));
    this.agents.set("frontend", new FrontendAgent(agentConfigs.frontend));
    this.agents.set("backend", new BackendAgent(agentConfigs.backend));
    this.agents.set("database", new DatabaseAgent(agentConfigs.database));
    this.agents.set("devops", new DevOpsAgent(agentConfigs.devops));
    this.agents.set("qa", new QAAgent(agentConfigs.qa));
    this.agents.set("security", new SecurityAgent(agentConfigs.security));
    this.agents.set(
      "performance",
      new PerformanceAgent(agentConfigs.performance)
    );
    this.agents.set(
      "documentation",
      new DocumentationAgent(agentConfigs.documentation)
    );
    this.agents.set(
      "project_manager",
      new ProjectManagerAgent(agentConfigs.project_manager)
    );
  }

  private buildWorkflow(): void {
    this.workflow = new StateGraph(WorkflowState)
      // Analysis and Planning Phase
      .addNode("project_analysis", this.analyzeProject.bind(this))
      .addNode("architecture_design", this.designArchitecture.bind(this))
      .addNode("task_planning", this.planTasks.bind(this))
      .addNode("agent_assignment", this.assignAgents.bind(this))

      // Development Phase
      .addNode("parallel_development", this.parallelDevelopment.bind(this))
      .addNode("code_review", this.conductCodeReview.bind(this))
      .addNode("integration", this.integrateComponents.bind(this))

      // Quality Assurance Phase
      .addNode("testing", this.executeTests.bind(this))
      .addNode("security_audit", this.conductSecurityAudit.bind(this))
      .addNode("performance_optimization", this.optimizePerformance.bind(this))

      // Deployment Phase
      .addNode("deployment_preparation", this.prepareDeployment.bind(this))
      .addNode("deployment", this.deployProject.bind(this))
      .addNode("monitoring_setup", this.setupMonitoring.bind(this))

      // Maintenance Phase
      .addNode("documentation", this.generateDocumentation.bind(this))
      .addNode("handover", this.projectHandover.bind(this))

      // Support Nodes
      .addNode("blocker_resolution", this.resolveBlockers.bind(this))
      .addNode("quality_gate", this.checkQualityGate.bind(this))
      .addNode("stakeholder_update", this.updateStakeholders.bind(this));

    // Define the workflow flow
    this.workflow
      .addEdge(START, "project_analysis")
      .addEdge("project_analysis", "architecture_design")
      .addEdge("architecture_design", "task_planning")
      .addEdge("task_planning", "agent_assignment")
      .addEdge("agent_assignment", "parallel_development")

      // Conditional edges for development flow
      .addConditionalEdges(
        "parallel_development",
        this.checkDevelopmentStatus.bind(this),
        {
          ready_for_review: "code_review",
          blocked: "blocker_resolution",
          continue: "parallel_development",
          quality_check: "quality_gate",
        }
      )

      .addConditionalEdges("code_review", this.checkReviewStatus.bind(this), {
        approved: "integration",
        needs_changes: "parallel_development",
        blocked: "blocker_resolution",
      })

      .addConditionalEdges(
        "integration",
        this.checkIntegrationStatus.bind(this),
        {
          success: "testing",
          conflicts: "parallel_development",
          blocked: "blocker_resolution",
        }
      )

      .addConditionalEdges("testing", this.checkTestResults.bind(this), {
        passed: "security_audit",
        failed: "parallel_development",
        blocked: "blocker_resolution",
      })

      .addEdge("security_audit", "performance_optimization")
      .addEdge("performance_optimization", "deployment_preparation")
      .addEdge("deployment_preparation", "deployment")
      .addEdge("deployment", "monitoring_setup")
      .addEdge("monitoring_setup", "documentation")
      .addEdge("documentation", "handover")
      .addEdge("handover", END)

      // Blocker resolution flows back to appropriate phases
      .addConditionalEdges(
        "blocker_resolution",
        this.determineReturnPhase.bind(this),
        {
          development: "parallel_development",
          review: "code_review",
          testing: "testing",
          deployment: "deployment_preparation",
        }
      )

      // Quality gate can route to different phases
      .addConditionalEdges(
        "quality_gate",
        this.checkQualityMetrics.bind(this),
        {
          passed: "code_review",
          failed: "parallel_development",
          needs_optimization: "performance_optimization",
        }
      );
  }

  // Workflow node implementations
  private async analyzeProject(
    state: typeof WorkflowState.State
  ): Promise<typeof WorkflowState.State> {
    const projectManager = this.agents.get("project_manager");
    const architect = this.agents.get("architect");

    // Project Manager analyzes requirements
    const pmAnalysis = await projectManager.analyzeRequirements({
      project_id: state.project_id,
      context: state.project_context,
    });

    // Architect provides technical feasibility assessment
    const techAnalysis = await architect.assessTechnicalFeasibility({
      requirements: pmAnalysis.requirements,
      constraints: pmAnalysis.constraints,
    });

    return {
      ...state,
      current_phase: "analysis",
      project_context: {
        ...state.project_context,
        requirements: pmAnalysis.requirements,
        constraints: pmAnalysis.constraints,
        feasibility: techAnalysis.feasibility,
        risks: [...pmAnalysis.risks, ...techAnalysis.risks],
      },
      decisions: [
        ...state.decisions,
        {
          type: "analysis_complete",
          timestamp: new Date().toISOString(),
          details: { pmAnalysis, techAnalysis },
        },
      ],
    };
  }

  private async designArchitecture(
    state: typeof WorkflowState.State
  ): Promise<typeof WorkflowState.State> {
    const architect = this.agents.get("architect");
    const database = this.agents.get("database");
    const security = this.agents.get("security");

    // Architect designs system architecture
    const architecture = await architect.designSystemArchitecture({
      requirements: state.project_context.requirements,
      constraints: state.project_context.constraints,
    });

    // Database agent designs data architecture
    const dataArchitecture = await database.designDataArchitecture({
      requirements: state.project_context.requirements,
      system_architecture: architecture.components,
    });

    // Security agent defines security requirements
    const securityRequirements = await security.defineSecurityRequirements({
      architecture: architecture,
      data_flows: dataArchitecture.flows,
    });

    return {
      ...state,
      current_phase: "design",
      project_context: {
        ...state.project_context,
        architecture: architecture,
        data_architecture: dataArchitecture,
        security_requirements: securityRequirements,
      },
      decisions: [
        ...state.decisions,
        {
          type: "architecture_complete",
          timestamp: new Date().toISOString(),
          details: { architecture, dataArchitecture, securityRequirements },
        },
      ],
    };
  }

  private async planTasks(
    state: typeof WorkflowState.State
  ): Promise<typeof WorkflowState.State> {
    const projectManager = this.agents.get("project_manager");

    const taskPlan = await projectManager.createDetailedTaskPlan({
      project_context: state.project_context,
      available_agents: Array.from(this.agents.keys()),
    });

    return {
      ...state,
      current_phase: "planning",
      active_tasks: taskPlan.tasks,
      timeline: taskPlan.timeline,
      agent_assignments: {},
      decisions: [
        ...state.decisions,
        {
          type: "task_planning_complete",
          timestamp: new Date().toISOString(),
          details: taskPlan,
        },
      ],
    };
  }

  private async assignAgents(
    state: typeof WorkflowState.State
  ): Promise<typeof WorkflowState.State> {
    const projectManager = this.agents.get("project_manager");

    const assignments = await projectManager.assignAgentsToTasks({
      tasks: state.active_tasks,
      agents: Array.from(this.agents.entries()).map(([id, agent]) => ({
        id,
        specialization: agent.getSpecialization(),
        capabilities: agent.getCapabilities(),
        current_load: agent.getCurrentLoad(),
      })),
    });

    // Notify agents of their assignments
    for (const [taskId, agentId] of Object.entries(assignments.assignments)) {
      const agent = this.agents.get(agentId);
      const task = state.active_tasks.find((t) => t.id === taskId);
      if (agent && task) {
        await agent.assignTask(task);
      }
    }

    return {
      ...state,
      current_phase: "development",
      agent_assignments: assignments.assignments,
      decisions: [
        ...state.decisions,
        {
          type: "agent_assignment_complete",
          timestamp: new Date().toISOString(),
          details: assignments,
        },
      ],
    };
  }

  private async parallelDevelopment(
    state: typeof WorkflowState.State
  ): Promise<typeof WorkflowState.State> {
    const developmentPromises = [];
    const newArtifacts = [];
    const updatedTasks = [...state.active_tasks];

    // Execute tasks in parallel for different agents
    for (const [taskId, agentId] of Object.entries(state.agent_assignments)) {
      const agent = this.agents.get(agentId);
      const task = state.active_tasks.find((t) => t.id === taskId);

      if (agent && task && task.status === "assigned") {
        developmentPromises.push(
          agent
            .executeTask(task, {
              project_context: state.project_context,
              related_artifacts: state.artifacts.filter(
                (a) => a.task_id === taskId
              ),
            })
            .then((result) => ({ taskId, agentId, result }))
        );
      }
    }

    // Wait for all parallel development to complete
    const results = await Promise.allSettled(developmentPromises);

    // Process results
    for (const promiseResult of results) {
      if (promiseResult.status === "fulfilled") {
        const { taskId, agentId, result } = promiseResult.value;

        // Add artifacts
        newArtifacts.push(...result.artifacts);

        // Update task status
        const taskIndex = updatedTasks.findIndex((t) => t.id === taskId);
        if (taskIndex >= 0) {
          updatedTasks[taskIndex] = {
            ...updatedTasks[taskIndex],
            status: result.status,
            progress_percentage: result.progress_percentage,
            actual_hours: result.actual_hours,
            completed_at:
              result.status === "completed" ? new Date().toISOString() : null,
          };
        }
      }
    }

    return {
      ...state,
      active_tasks: updatedTasks,
      artifacts: [...state.artifacts, ...newArtifacts],
      decisions: [
        ...state.decisions,
        {
          type: "development_iteration_complete",
          timestamp: new Date().toISOString(),
          details: { completed_tasks: results.length },
        },
      ],
    };
  }

  // Status checking methods for conditional edges
  private async checkDevelopmentStatus(
    state: typeof WorkflowState.State
  ): Promise<string> {
    const inProgressTasks = state.active_tasks.filter(
      (t) => t.status === "in_progress"
    );
    const blockedTasks = state.active_tasks.filter(
      (t) => t.status === "blocked"
    );
    const completedTasks = state.active_tasks.filter(
      (t) => t.status === "completed"
    );
    const reviewTasks = state.active_tasks.filter(
      (t) => t.status === "ready_for_review"
    );

    if (blockedTasks.length > 0) {
      return "blocked";
    }

    if (reviewTasks.length > 0) {
      return "ready_for_review";
    }

    if (inProgressTasks.length === 0 && completedTasks.length > 0) {
      // Check if quality gates need to be evaluated
      const needsQualityCheck = await this.needsQualityCheck(state);
      return needsQualityCheck ? "quality_check" : "ready_for_review";
    }

    return "continue";
  }

  private async checkReviewStatus(
    state: typeof WorkflowState.State
  ): Promise<string> {
    // Implementation for code review status checking
    const reviewResults = await this.getLatestReviewResults(state);

    if (reviewResults.blockers.length > 0) {
      return "blocked";
    }

    if (reviewResults.changesRequired) {
      return "needs_changes";
    }

    return "approved";
  }

  // Additional helper methods...

  async startProject(projectId: string): Promise<void> {
    const initialState = {
      project_id: projectId,
      current_phase: "analysis",
      active_tasks: [],
      agent_assignments: {},
      project_context: await this.loadProjectContext(projectId),
      decisions: [],
      artifacts: [],
      blockers: [],
      quality_metrics: {},
      timeline: {},
      next_actions: [],
    };

    const app = this.workflow.compile();

    for await (const output of await app.stream(initialState)) {
      console.log("Workflow step completed:", output);
      await this.saveWorkflowState(projectId, output);
    }
  }

  private async loadProjectContext(
    projectId: string
  ): Promise<Record<string, any>> {
    // Load project requirements and context from database
    // Implementation details...
    return {};
  }

  private async saveWorkflowState(
    projectId: string,
    state: any
  ): Promise<void> {
    // Save current workflow state to database
    // Implementation details...
  }
}
```

I'll continue with the agent implementations and prompt scaffolding. Would you like me to continue with the complete DEVAO implementation including all 10 agents, their specific prompt templates, and usage examples?
