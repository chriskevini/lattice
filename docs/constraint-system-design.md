# Resource Constraint System Design

## Problem Statement

In an AI system with hardware constraints (2GB RAM / 1vCPU), how do we balance:
1. AI autonomy to request optimal context for generating the best response
2. System stability and performance requirements
3. Transparency so AI understands when it's being constrained

## Core Philosophy

**AI's Goal**: Generate the best possible response
**System's Goal**: Enforce hardware constraints to prevent crashes

The AI should not be asked to "care about" system efficiency. That's the system's job. Instead:
- AI analyzes conversation needs and requests appropriate context
- System transparently enforces limits and reports back
- AI learns what's possible and adapts strategies over time

## The Core Dilemma

**Without transparency**:
- AI requests 20 turns of context
- System silently caps at 15 turns
- AI generates response with insufficient context
- AI doesn't know why responses are suboptimal

**With preset "strategies"** (rejected approach):
- System offers "minimal/balanced/comprehensive" presets
- AI is forced to think about system efficiency instead of response quality
- False abstraction that doesn't match AI's actual decision-making process
- Removes AI agency to make context decisions based on conversation needs

**With full autonomy**:
- AI requests unlimited context
- System runs out of memory and crashes
- User experience is destroyed

## Solution: Direct Resource Requests with Transparent Clamping

### How Context Needs Vary

The same AI might need very different context depending on the conversation:

**Example 1: Deep Technical Debugging**
```
Conversation: User debugging specific function mentioned 8 turns ago
AI Analysis: Need full conversation thread, minimal semantic noise
Request: CONTEXT_TURNS: 15, VECTOR_LIMIT: 2, SIMILARITY_THRESHOLD: 0.8
Result: Deep sequential context, tightly focused semantic search
```

**Example 2: Broad Preference Recall**
```
Conversation: "What are my favorite hobbies?"
AI Analysis: User preferences span many topics and time periods
Request: CONTEXT_TURNS: 3, VECTOR_LIMIT: 12, SIMILARITY_THRESHOLD: 0.6
Result: Recent context + wide semantic search
```

**Example 3: Simple Continuation**
```
Conversation: "Thanks!" (responding to previous answer)
AI Analysis: Everything needed is in immediate context
Request: CONTEXT_TURNS: 2, VECTOR_LIMIT: 0
Result: Minimal resources, fast response
```

**Example 4: Relational Reasoning**
```
Conversation: "What projects might I enjoy based on my interests?"
AI Analysis: Need to traverse relationships between interests
Request: CONTEXT_TURNS: 5, VECTOR_LIMIT: 6, TRIPLE_DEPTH: 3
Result: Recent context + semantic search + multi-hop graph traversal
```

### Design Principles

1. **AI decides context needs**: Based on conversation analysis, not system constraints
2. **Ranges, not fixed limits**: Min/Max/Default for each resource dimension
3. **Multiple resource dimensions**: Turns, vectors, similarity, graph depth
4. **Transparent clamping**: System reports when requests exceed limits
5. **Learning through feedback**: AI discovers optimal patterns within constraints
6. **Evolution mechanism**: AI can propose limit adjustments via dreaming cycle

### Resource Dimensions

AI can independently control multiple context dimensions:

1. **CONTEXT_TURNS** (1-20): Sequential conversation history
   - More turns = better thread continuity, temporal context
   - Use when: Following long discussions, callbacks to earlier topics

2. **VECTOR_LIMIT** (0-15): Semantic similarity search results
   - More vectors = broader knowledge recall, cross-domain connections
   - Use when: Exploring preferences, broad topics, unknown territory

3. **SIMILARITY_THRESHOLD** (0.5-0.9): How strict semantic matching is
   - Higher = more relevant but narrower results
   - Lower = broader but potentially noisier results
   - Use when: Adjusting semantic search precision

4. **TRIPLE_DEPTH** (0-3): How many relationship hops to traverse
   - More depth = multi-step reasoning (A→B→C)
   - Use when: Inferring indirect connections, complex reasoning

### Implementation

#### 1. Configuration (.env)

```bash
# Philosophy: AI requests what it needs, system enforces limits
MIN_EPISODIC_CONTEXT_TURNS=1
MAX_EPISODIC_CONTEXT_TURNS=20
DEFAULT_EPISODIC_CONTEXT_TURNS=10

MIN_VECTOR_SEARCH_LIMIT=0
MAX_VECTOR_SEARCH_LIMIT=15
DEFAULT_VECTOR_SEARCH_LIMIT=5

MIN_SIMILARITY_THRESHOLD=0.5
MAX_SIMILARITY_THRESHOLD=0.9
DEFAULT_SIMILARITY_THRESHOLD=0.7

MIN_TRIPLE_DEPTH=0
MAX_TRIPLE_DEPTH=3
DEFAULT_TRIPLE_DEPTH=1

# Transparency
REPORT_CONSTRAINT_VIOLATIONS=true
INCLUDE_EXECUTION_REPORT=true
```

#### 2. AI Request Interface (in prompt_registry template)

```
Based on this conversation, determine optimal context retrieval:

CONTEXT_TURNS: <1-20> (sequential conversation history)
VECTOR_LIMIT: <0-15> (semantic search results)  
SIMILARITY_THRESHOLD: <0.5-0.9> (semantic matching strictness)
TRIPLE_DEPTH: <0-3> (relationship graph hops)

Examples:
- Deep technical thread: CONTEXT_TURNS:15, VECTOR_LIMIT:2, SIMILARITY_THRESHOLD:0.8
- Broad preference recall: CONTEXT_TURNS:3, VECTOR_LIMIT:12, SIMILARITY_THRESHOLD:0.6
- Simple continuation: CONTEXT_TURNS:2, VECTOR_LIMIT:0
- Complex reasoning: CONTEXT_TURNS:5, VECTOR_LIMIT:6, TRIPLE_DEPTH:3

Hardware constraints (2GB RAM / 1vCPU):
- CONTEXT_TURNS: 1-20 (default: 10)
- VECTOR_LIMIT: 0-15 (default: 5)
- SIMILARITY_THRESHOLD: 0.5-0.9 (default: 0.7)
- TRIPLE_DEPTH: 0-3 (default: 1)

Requests outside ranges will be clamped and you'll receive a report.
```

#### 3. Execution Reporting

When AI requests are clamped, system includes in next context:

```
SYSTEM_EXECUTION_REPORT:
Your previous request:
  CONTEXT_TURNS: 25
  VECTOR_LIMIT: 18
  SIMILARITY_THRESHOLD: 0.85
  TRIPLE_DEPTH: 2

Actual resources provided:
  CONTEXT_TURNS: 20 (clamped to MAX_EPISODIC_CONTEXT_TURNS)
  VECTOR_LIMIT: 15 (clamped to MAX_VECTOR_SEARCH_LIMIT)
  SIMILARITY_THRESHOLD: 0.85 (as requested)
  TRIPLE_DEPTH: 2 (as requested)

Constraint reason: Hardware limit (2GB RAM / 1vCPU)
Performance metrics: Memory: 1.6GB/2.0GB (80%), Query time: 520ms

Suggestion: Your request for 25 turns suggests you need deep thread context.
Consider using semantic triples (TRIPLE_DEPTH) to reconstruct earlier 
conversation context beyond the 20-turn limit.
```

This report:
- Shows what AI requested vs what it got
- Explains why (hardware constraint)
- Provides performance context
- Suggests alternatives within constraints

#### 4. Dreaming Cycle Proposals

AI can propose constraint adjustments with evidence:

```
CONSTRAINT_ADJUSTMENT_PROPOSAL:
Resource: MAX_EPISODIC_CONTEXT_TURNS
Current: 20
Proposed: 25
Rationale: 
  Analysis of 100 recent conversations shows:
  - 15 conversations hit the 20-turn limit
  - In 12 cases, user asked follow-up questions suggesting missing context
  - Average turn count when limit hit: 23 (suggesting natural need for ~25)
  
Risk Assessment:
  Memory impact: +250MB per query (tested in simulation)
  Performance impact: +100ms avg query time
  Peak memory: 1.85GB (safe headroom from 2GB limit)
  Crash risk: LOW
  
Evidence:
  - Conversation #1234: User debugging, needed reference to turn 22
  - Conversation #1267: Long planning discussion truncated
  - User satisfaction scores: 8.2/10 when hitting limit vs 9.1/10 otherwise
  
Alternative considered: Using TRIPLE_DEPTH to reconstruct old context
  Result: Less effective - loses conversational flow and exact wording

Proposed test: Increase limit to 25 for 1 week, monitor memory/performance
```

Human reviews in Dream Channel and approves/rejects with reasoning.

## Benefits

### For AI
- **Awareness**: Knows constraints exist and what they are
- **Learning**: Discovers optimal resource usage over time
- **Agency**: Can propose changes with evidence
- **Understanding**: Gets feedback on why requests were modified

### For System
- **Stability**: Hard limits prevent crashes
- **Performance**: Resources stay within budget
- **Adaptability**: Can tune constraints based on AI proposals
- **Observability**: Logs show AI's resource usage patterns

### For Users
- **Reliability**: System doesn't crash
- **Quality**: AI learns to use resources optimally
- **Transparency**: Can see AI's reasoning in Dream Channel
- **Evolution**: System improves over time

## Edge Cases

### AI Never Sets Proactive Interval
- **Failsafe**: `DEFAULT_PROACTIVE_INTERVAL_MINUTES=120`
- System continues functioning with reasonable default

### AI Always Requests Maximum
- **Feedback loop**: Execution reports show performance impact
- **Dreaming cycle**: AI analyzes if max is actually needed
- **Cost awareness**: Prompt templates can include performance context

### AI Proposes Unsafe Constraint
- **Human approval gate**: All proposals go to Dream Channel
- **Evidence required**: AI must provide rationale and risk assessment
- **Testing**: Can be tested in staging before production

### Resource Availability Changes
- **Dynamic adjustment**: `.env` can be updated without code changes
- **Notification**: AI is informed of new constraints in next context
- **Re-proposal**: AI can re-propose previously rejected changes

## Implementation Priority

### Phase 1: Basic Constraints (MVP)
- [x] Define ranges in `.env`
- [ ] Implement clamping logic
- [ ] Basic logging of clamped requests

### Phase 2: Transparency
- [ ] Execution reports in context
- [ ] Multi-dimensional request parsing (CONTEXT_TURNS, VECTOR_LIMIT, etc.)
- [ ] Performance metrics in reports

### Phase 3: Evolution
- [ ] Dreaming cycle proposals
- [ ] Human approval workflow
- [ ] Constraint update mechanism
- [ ] Historical analysis of resource usage

## Alternative Approaches Considered

### 1. Fixed Limits (Rejected)
**Pros**: Simple to implement
**Cons**: No AI awareness, no learning, no adaptation

### 2. Unlimited Resources (Rejected)
**Pros**: Maximum AI autonomy
**Cons**: System crashes, terrible UX, violates hardware constraints

### 3. Preset Strategies: minimal/balanced/comprehensive (Rejected)
**Pros**: Simple abstraction for AI to choose from
**Cons**: 
- AI doesn't care about system efficiency - its goal is best response
- Removes AI agency to make nuanced context decisions
- False abstraction that doesn't match actual decision-making needs
- Same AI might need turn-heavy context for debugging but vector-heavy for preferences

### 4. External API Rate Limiting Patterns (Rejected)
**Pros**: Industry standard approach
**Cons**: Not appropriate for local hardware constraints (different problem domain)

### 5. Cost-Based System (Deferred)
**Pros**: AI learns resource "prices", makes economic trade-offs
**Cons**: Too complex for MVP, harder for AI to understand than direct ranges

## Conclusion

The Resource Constraint System balances AI autonomy with system stability through:

1. **AI-centric design**: AI requests what it needs for best response, not what's "efficient"
2. **Multiple resource dimensions**: Independent control of turns, vectors, similarity, graph depth
3. **Transparent ranges**: AI knows limits and gets feedback when clamped
4. **Execution reporting**: AI learns from experience what works within constraints
5. **Evolution mechanism**: AI proposes evidence-based improvements
6. **Human oversight**: Safety gate for constraint changes

This approach treats the AI as a collaborative partner in resource management, empowering it to make context decisions based on conversation needs while the system handles stability concerns. The AI doesn't need to "care about" performance - that's the system's job.
