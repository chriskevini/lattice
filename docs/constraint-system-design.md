# Resource Constraint System Design

## Problem Statement

In an AI system with hardware constraints (2GB RAM / 1vCPU), how do we balance:
1. AI autonomy to request optimal context
2. System stability and performance requirements
3. Transparency so AI understands when it's being constrained

## The Core Dilemma

**Without transparency**:
- AI requests 20 turns of context
- System silently caps at 10 turns
- AI generates response with insufficient context
- AI doesn't know why responses are suboptimal

**With naive transparency**:
- AI is told "max 10 turns"
- AI always requests 10 turns (anchoring bias)
- No learning or adaptation occurs

**With full autonomy**:
- AI requests unlimited context
- System runs out of memory and crashes
- User experience is destroyed

## Solution: Adaptive Constraints with Feedback

### Design Principles

1. **Ranges, not fixed limits**: Min/Max/Default for each resource
2. **Strategy-based requests**: AI thinks in outcomes, not implementation
3. **Execution reporting**: System tells AI what it actually got
4. **Learning over time**: AI discovers optimal resource usage through feedback
5. **Dreaming cycle adjustments**: AI can propose constraint changes with rationale

### Implementation

#### 1. Configuration (.env)

```bash
# Each resource has a range
MIN_EPISODIC_CONTEXT_TURNS=3
MAX_EPISODIC_CONTEXT_TURNS=15
DEFAULT_EPISODIC_CONTEXT_TURNS=10

# Transparency flags
REPORT_CONSTRAINT_VIOLATIONS=true
INCLUDE_EXECUTION_REPORT=true
```

#### 2. AI Request Interface

AI can request resources in two ways:

**A. High-level Strategy** (Recommended):
```
RETRIEVAL_STRATEGY: comprehensive
```

System translates:
- `minimal` → 3 turns, 3 vectors (fast)
- `balanced` → 10 turns, 5 vectors (default)
- `comprehensive` → 15 turns, 10 vectors (thorough)

**B. Explicit Values**:
```
CONTEXT_TURNS: 12
VECTOR_LIMIT: 8
```

System clamps to ranges if needed.

#### 3. Execution Reporting

When AI requests are clamped, system includes in next context:

```
SYSTEM_EXECUTION_REPORT:
Previous request: 20 context turns, 12 vector results
Actual provided: 15 context turns (MAX), 10 vector results (MAX)
Reason: Hardware constraint (2GB RAM / 1vCPU)
Suggestion: Use semantic search for historical context beyond 15 turns
Memory usage: 1.4GB / 2.0GB (70%)
Query time: 450ms
```

#### 4. Dreaming Cycle Proposals

AI can propose constraint adjustments:

```
CONSTRAINT_ADJUSTMENT_PROPOSAL:
Resource: MAX_VECTOR_SEARCH_LIMIT
Current: 10
Proposed: 12
Rationale: Analysis of 50 recent conversations shows frequent need for 
broader context. User topics span multiple domains requiring more 
semantic connections.
Risk assessment: 
  - Memory impact: +15% per query (tested)
  - Performance impact: +80ms avg (acceptable)
  - Crash risk: LOW (peak usage would be 1.8GB with headroom)
Evidence: 
  - 23 conversations truncated useful results
  - User gave positive feedback when comprehensive strategy used
```

Human approves/rejects in Dream Channel.

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
- [ ] Strategy-based request system
- [ ] Performance metrics in reports

### Phase 3: Evolution
- [ ] Dreaming cycle proposals
- [ ] Human approval workflow
- [ ] Constraint update mechanism
- [ ] Historical analysis of resource usage

## Alternative Approaches Considered

### 1. Fixed Limits (Rejected)
**Pros**: Simple
**Cons**: No AI awareness, no learning, no adaptation

### 2. Unlimited Resources (Rejected)
**Pros**: Maximum AI autonomy
**Cons**: System crashes, terrible UX

### 3. External API Rate Limiting (Rejected)
**Pros**: Industry standard
**Cons**: Not appropriate for local hardware constraints

### 4. Cost-Based System (Deferred)
**Pros**: AI learns resource "prices", makes economic decisions
**Cons**: Too complex for MVP, hard for AI to understand

## Conclusion

The Resource Constraint System balances AI autonomy with system stability through:
1. Transparent ranges (not fixed limits)
2. Execution reporting (AI learns from experience)
3. Evolution mechanism (AI proposes improvements)
4. Human oversight (safety gate for changes)

This approach treats the AI as a collaborative partner in resource management, rather than either giving unlimited control or hiding constraints entirely.
