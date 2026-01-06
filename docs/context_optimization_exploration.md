# Context Optimization Exploration

## Problem Statement

### Background: What is Lattice?
Lattice is a Discord companion bot with a memory system running on constrained resources (2GB RAM, free LLM tier). When a user sends a message, the bot:
1. Stores the message in a database
2. Retrieves relevant context (past conversations, stored facts)
3. Generates a response using an LLM (Large Language Model)

**Key Constraint**: Uses free LLM models, so **response quality and accuracy matter more than token costs**.

### The Core Problem: One-Size-Fits-All Context Retrieval
Currently, **every message type gets the same amount of context**, regardless of whether it's needed:

```python
# Current code - hardcoded for all messages:
context = retrieve_context(
    episodic_limit=10,      # Last 10 Discord messages
    triple_depth=1,         # 1 hop in knowledge graph
)
```

This creates two issues:

#### Issue 1: Quality Degradation
Different message types need different amounts of context:

| User Message | Type | Context Needed | Current Context | Problem |
|--------------|------|----------------|-----------------|---------|
| "Spent 3 hours coding" | Activity update | None (self-contained) | 10 messages + graph | Unnecessary noise in prompt |
| "What did I work on last week?" | Query | Deep knowledge graph | 10 messages + shallow graph | **Missing connections = wrong answers** |
| "lol yeah that's true" | Conversation | Recent chat flow | 10 messages + graph | Graph pollutes conversational flow |

**Primary Impact**: Suboptimal responses due to wrong context mix  
**Secondary Impact**: Unnecessary latency (~100-200ms per message) from retrieving/processing unused context

#### Issue 2: Inconsistent Bot Personality
- System uses **5 different response templates** (QUERY_RESPONSE, ACTIVITY_RESPONSE, etc.)
- Each template has different instructions and tone
- Result: Bot voice changes depending on message classification
- Users perceive this as inconsistent or "split personality"

### Context Types Explained

For readers unfamiliar with the system, here's what "context" means:

**Episodic Context** (Recent conversation messages):
```
[2026-01-05 14:23] USER: I need to finish the lattice project by Friday
[2026-01-05 14:24] BOT: Got it! Friday deadline for lattice.
[2026-01-05 14:30] USER: Just spent 3 hours coding
[2026-01-05 14:31] BOT: Nice session! How'd it go?
```
- Source: Last N messages from Discord chat history
- Token cost: ~30-50 tokens per message
- Use case: Understanding conversation flow

**Graph Context** (Knowledge relationships):
```
Related knowledge:
- lattice project has_deadline Friday Jan 10th
- lattice project uses_technology PostgreSQL  
- user working_on lattice project
- PostgreSQL requires asyncpg library
- Friday date_in_month January
```
- Source: Knowledge graph with entities and relationships
- Token cost: ~15-25 tokens per triple
- Graph depth controls how many "hops" to follow:
  - Depth 0: No graph (0 tokens)
  - Depth 1: Direct connections (~50-150 tokens, 3-6 triples)
  - Depth 2: Two hops (~200-400 tokens, 8-16 triples)
- Use case: Answering factual questions, connecting past information

### Current LLM Call Flow

Understanding the solution requires knowing the current architecture:

```
User Message: "Spent 3 hours coding"
    ↓
1. Store message in database
    ↓
2. Query Extraction (LLM Call #1)
   Input: Message + last 3 messages for context
   Output: {
     message_type: "activity_update",
     entities: ["coding"],
     predicates: ["spent_time"],
     ...
   }
    ↓
3. Retrieve Context (Database queries)
   - Fetch 10 recent messages (episodic)
   - Traverse knowledge graph depth=1 (semantic)
    ↓
4. Generate Response (LLM Call #2)
   Template: Select based on message_type
   Templates: QUERY_RESPONSE, ACTIVITY_RESPONSE,
              CONVERSATION_RESPONSE, GOAL_RESPONSE, etc.
   Input: Message + full context + template instructions
   Output: Bot's response
```

**Key insight**: System already makes **two LLM calls** per message. The first call (extraction) is cheap (~50-200 tokens), the second call (response) is expensive (~1500-3000 tokens including context).

---

## Design A: Static Context Limits per Template

### Core Idea
Different message types have different context needs. Set **static, sensible defaults** for each template type.

### How It Works

**1. Add context configuration to database:**
```sql
ALTER TABLE prompt_registry
ADD COLUMN context_limits JSONB DEFAULT '{
  "episodic_limit": 10,
  "triple_depth": 1,
  "max_graph_triples": 10
}'::jsonb;
```

**2. Configure limits per template:**
```json
{
  "QUERY_RESPONSE": {
    "episodic_limit": 6,        // Fewer messages (queries about past, not present)
    "triple_depth": 2,           // Deeper graph (need connections to answer)
    "max_graph_triples": 15
  },
  "ACTIVITY_RESPONSE": {
    "episodic_limit": 5,         // Minimal messages
    "triple_depth": 0,           // No graph needed (self-contained)
    "max_graph_triples": 0
  },
  "CONVERSATION_RESPONSE": {
    "episodic_limit": 12,        // More messages (conversation flow matters)
    "triple_depth": 0,           // No graph (casual chat)
    "max_graph_triples": 0
  },
  "GOAL_RESPONSE": {
    "episodic_limit": 8,         // Balanced
    "triple_depth": 1,           // Some connections useful
    "max_graph_triples": 12
  }
}
```

**3. Modified flow:**
```
User Message
    ↓
1. Query Extraction → message_type = "activity_update"
    ↓
2. Select Template → ACTIVITY_RESPONSE
    ↓
3. Load Template Config → episodic_limit=5, triple_depth=0
    ↓
4. Retrieve Context (using template's limits)
    ↓
5. Generate Response
```

### Quality Impact Analysis

| Template | Current Context | Optimized Context | Quality Impact |
|----------|----------------|-------------------|----------------|
| ACTIVITY | 10 msgs + depth 1 | 5 msgs + depth 0 | ✅ **Better** - Less noise, clearer acknowledgment |
| CONVERSATION | 10 msgs + depth 1 | 12 msgs + depth 0 | ✅ **Better** - More conversational flow, no graph pollution |
| GOAL | 10 msgs + depth 1 | 8 msgs + depth 1 | ≈ **Similar** - Balanced context maintained |
| QUERY | 10 msgs + depth 1 | 6 msgs + depth 2 | ✅✅ **Much Better** - Deeper graph = more accurate answers |

**Key Benefit**: QUERY responses become significantly more accurate with depth=2 graph traversal, finding connections that depth=1 misses.

**Secondary Benefits:**
- ~21% token reduction (520 → 410 avg) - modest latency improvement
- More predictable context per message type
- Clearer prompt construction

### Pros
✅ Simple to implement and understand  
✅ Predictable context retrieval  
✅ Better quality for common message types (especially queries)  
✅ Works even if extraction fails (use BASIC_RESPONSE defaults)  
✅ Easy to tune based on user feedback  

### Cons
❌ Static rules can't adapt to edge cases  
❌ Requires manual tuning when patterns change  
❌ May over-provision or under-provision for ambiguous messages  
❌ **Still maintains 5+ separate templates** - inconsistent bot voice  
❌ Extraction still computes unused fields (message_type, entities, predicates, continuation)  

### Implementation Complexity
**Time estimate**: 2-3 hours

**Changes required**:
1. Database migration (add `context_limits` column)
2. Update `PromptTemplate` class to load limits
3. Update `bot.py` to pass template limits to retrieval
4. Add tests

**Risk**: Low - straightforward changes, easy to rollback

---

## Design B: LLM-Driven Context Requests (Radical Approach)

### Core Idea
The LLM analyzing the message **knows best** what context it needs. Let it **request** the context, then provide exactly what was asked for. This enables:

1. **Adaptive context** - Right amount for each unique message
2. **Unified template** - Single consistent bot voice
3. **Optimal quality** - LLM gets precisely what it needs to answer well

### How It Works

**1. Enhanced extraction output:**

Current extraction output:
```json
{
  "message_type": "query",
  "entities": ["lattice project"],
  "predicates": ["worked_on"],
  "continuation": false
}
```

Enhanced extraction output:
```json
{
  "message_type": "query",
  "entities": ["lattice project"],
  "predicates": ["worked_on"],
  "continuation": false,

  // NEW: Context requests
  "request": {
    "more_history": true,     // Need more conversation messages?
    "more_triples": true      // Need knowledge graph?
  }
}
```

**2. Modified flow:**
```
User Message: "What did I work on last week?"
    ↓
1. Query Extraction (LLM Call #1)
   Input: Message + last 10 messages
   LLM thinks: "This is asking about past work. I don't see
                that info in recent chat, need graph data."
   Output: {
     message_type: "query",
     entities: ["work", "last week"],
     request: {
       more_history: false,    // Recent chat won't help
       more_triples: true      // Need to find past activities
     }
   }
    ↓
2. Conditional Context Retrieval:
   IF more_history AND more_triples:
     → 20 messages + depth 2 graph
   ELSE IF more_history:
     → 20 messages + depth 0
   ELSE IF more_triples:
     → 10 messages + depth 2 graph
   ELSE:
     → 10 messages + depth 0
    ↓
3. Generate Response (LLM Call #2)
   Template: UNIFIED_RESPONSE (single template for all types)
   Input: Message + requested context
   Output: Bot's response
```

**3. Examples of LLM decision-making:**

| User Message | LLM Analysis | Context Request | Why |
|--------------|--------------|-----------------|-----|
| "Spent 3 hours coding" | Self-contained activity update | `{more_history: false, more_triples: false}` | Message says it all |
| "What's my Friday deadline?" | Query about stored fact | `{more_history: false, more_triples: true}` | Need graph to find deadline |
| "yeah that makes sense lol" | Conversational continuation | `{more_history: true, more_triples: false}` | Need chat context |
| "Did I finish that project I started in November?" | Deep query about past | `{more_history: true, more_triples: true}` | Need both chat history and graph |

### Quality Impact Analysis

**Design Philosophy**: **Prioritize accuracy over token efficiency** (using free models)

| Scenario | Frequency | Context Strategy | Quality Impact |
|----------|-----------|------------------|----------------|
| Self-contained (activity) | 40% | 10 msgs + depth 0 | ✅ **Good** - Minimal noise, clear acknowledgment |
| Conversational continuation | 30% | 20 msgs + depth 0 | ✅✅ **Excellent** - Full conversation context |
| Factual query | 20% | 10 msgs + depth 2 | ✅✅ **Excellent** - Deep graph finds answers |
| Complex query (past + context) | 10% | 20 msgs + depth 2 | ✅✅✅ **Outstanding** - Full context + deep graph |

**Key Advantages Over Design A:**

1. **Handles ambiguous messages** - LLM can request more context if unsure
2. **No misclassification penalty** - If "query" is mislabeled as "conversation", Design A gives wrong context; Design B adapts
3. **Conversation-aware queries** - Can request both history AND graph when needed (Design A forces one or the other)

**Example where Design B excels:**
```
User: "Did I ever finish that thing we talked about last week?"

Design A: Classifies as "query" → 6 messages + depth 2
Problem: Misses "that thing" reference in recent chat

Design B: Requests both → 20 messages + depth 2
Result: Finds both "that thing" from chat AND completion status from graph
```

### Additional Benefits

**1. Unified template = Coherent voice**
- Current: 5+ templates (QUERY_RESPONSE, ACTIVITY_RESPONSE, etc.) each with different instructions
- Design B: Single UNIFIED_RESPONSE template
- Result: Consistent personality across all message types

**2. Naturally adaptive**
- No manual tuning of static limits
- Automatically handles edge cases
- Self-adjusting to conversation patterns

**3. Simpler codebase**
- No template proliferation
- No complex routing logic
- Single response generation path

### Pros
✅ LLM makes intelligent decisions about context needs  
✅ Handles edge cases and ambiguous messages automatically  
✅ **Unified template = consistent bot personality**  
✅ No manual tuning required  
✅ Naturally adapts to conversation patterns  
✅ **Best quality for complex queries** (can request full context)  
✅ Conversation-aware query handling (can request both history + graph)

### Cons
❌ Higher token usage than Design A (16% more than current, but quality-focused)  
❌ More variability in token usage  
❌ Requires updating extraction schema  
❌ Extraction LLM must be reliable at context assessment  
❌ Still extracts unused fields (entities, predicates, continuation, etc.)  

### Implementation Complexity
**Time estimate**: 4-6 hours

**Changes required**:
1. Update `QueryExtraction` dataclass (add `request` fields)
2. Update `QUERY_EXTRACTION` template (teach LLM about context requests)
3. Update `bot.py` to conditionally retrieve context based on flags
4. Create new `UNIFIED_RESPONSE` template
5. Mark old response templates inactive (QUERY_RESPONSE, ACTIVITY_RESPONSE, etc.)
6. Add tests for conditional retrieval logic

**Risk**: Medium - changes core flow, requires careful testing of LLM request reliability

---

## Design C: Ultra-Simplified LLM Requests (Radical++)

### Core Insight: Current Extraction is Wasteful

**Discovery**: The current `QUERY_EXTRACTION` extracts many fields that **no response template actually uses**:

| Extraction Field | Used for Template Selection? | Used in Template Prompt? | Used for Anything Else? | Verdict |
|------------------|------------------------------|-------------------------|-------------------------|---------|
| `message_type` | ✅ Yes (routing) | ❌ No | ❌ No | **Only used for template selection** |
| `entities` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |
| `predicates` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |
| `continuation` | ❌ No | ❌ No | ❌ No | **Completely unused!** |
| `time_constraint` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |
| `activity` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |
| `query` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |
| `urgency` | ❌ No | ❌ No | ⚠️ Only for analytics | **Dead weight in extraction** |

**Current QUERY_EXTRACTION prompt**: 3,431 characters  
**Actual necessity**: Could be ~500 characters

**Note**: Consolidation (building knowledge graph) happens separately via `TRIPLE_EXTRACTION` after the response, so extraction entities aren't needed for that either.

### The Ultra-Simple Proposal

**If we use a unified template**, the extraction becomes trivially simple:

```json
{
  "more_history": true,   // Need extended conversation context?
  "more_triples": true    // Need knowledge graph data?
}
```

**That's it. Two boolean flags.**

### Minimal Extraction Prompt Example

```
User message: "{message}"

Recent conversation (last 10 messages):
{recent_messages}

Based on this message, what context do you need to generate a helpful response?

Output JSON:
{
  "more_history": true/false,  // Need more conversation history?
  "more_triples": true/false   // Need knowledge graph data?
}

Guidelines:
- more_history: true if message references conversation flow, pronouns like "that/it", or ongoing topics
- more_triples: true if asking about stored facts, past activities, deadlines, or personal information
- Both false for self-contained messages like simple greetings, activity updates, or clear statements

Examples:
- "Spent 3 hours coding" → {more_history: false, more_triples: false}
- "What's my Friday deadline?" → {more_history: false, more_triples: true}
- "yeah that makes sense" → {more_history: true, more_triples: false}
- "Did I finish that project from November?" → {more_history: true, more_triples: true}
```

**Prompt size**: ~500 chars (vs 3,431 current) = **85% reduction**

### Modified Flow

```
User Message: "What did I work on last week?"
    ↓
1. Context Request (LLM Call #1 - TINY)
   Input: Message + last 10 messages
   Output: {more_history: false, more_triples: true}
   Cost: ~100 tokens total
    ↓
2. Conditional Retrieval:
   more_triples=true → Fetch 10 msgs + depth 2 graph
    ↓
3. Unified Response (LLM Call #2)
   Single template for all message types
   Input: Message + requested context
   Output: Bot's response in consistent voice
   Cost: ~1500-3000 tokens
    ↓
4. Consolidation (async, unchanged)
   TRIPLE_EXTRACTION builds knowledge graph
   Cost: ~500 tokens
```

### Comparison to Design B

| Aspect | Design B (Original) | Design C (Ultra-Simple) |
|--------|-------------------|------------------------|
| Extraction fields | 8 fields (message_type, entities, etc.) | **2 fields** (boolean flags) |
| Extraction prompt | 3,431 chars | **~500 chars (85% smaller)** |
| Extraction tokens | ~200 tokens | **~50 tokens (75% cheaper)** |
| Template selection | Based on message_type | **No selection needed** |
| Unified template | ✅ Yes | ✅ Yes |
| Context adaptivity | ✅ Yes | ✅ Yes |
| Code complexity | Medium | **Lower** (fewer fields to handle) |

### Quality Impact

**Same quality as Design B**, but with benefits:

1. **Simpler extraction** → Less chance of extraction errors
2. **Faster extraction** → Lower latency for first LLM call
3. **Clearer prompt** → LLM has simpler decision to make
4. **Same adaptive context** → Still gets exactly what's needed

**Example decision-making:**

| User Message | LLM Analysis | Context Request | Result |
|--------------|--------------|-----------------|---------|
| "Spent 3 hours coding" | Self-contained statement | `{history: false, triples: false}` | Minimal context, quick response |
| "What's the deadline?" | Query about stored fact | `{history: false, triples: true}` | Deep graph search finds deadline |
| "yeah exactly" | Conversational continuation | `{history: true, triples: false}` | Extended chat context |
| "Did I finish that project I mentioned last month?" | Complex query referencing past | `{history: true, triples: true}` | Full context for accurate answer |

### Pros
✅ **Simplest possible extraction** (2 boolean flags)  
✅ **85% smaller extraction prompt** → faster, more reliable  
✅ **Unified template** = consistent bot personality  
✅ **Same quality benefits as Design B**  
✅ Less code complexity (fewer fields to manage)  
✅ Easier to debug (only 2 decisions to verify)  
✅ **Better extraction accuracy** (simpler decision = fewer errors)

### Cons
❌ Loses analytics fields (entities, predicates, etc.) - but these aren't used anyway  
❌ Same token usage variability as Design B  
❌ Still requires extraction LLM to be reliable (but simpler task)  
⚠️ If we later want to use extraction fields for something, need to add them back

### Implementation Complexity
**Time estimate**: 3-4 hours (simpler than Design B!)

**Changes required**:
1. Create new minimal `CONTEXT_REQUEST` template (~500 chars)
2. Update `QueryExtraction` dataclass (just 2 boolean fields)
3. Update `bot.py` to conditionally retrieve based on flags
4. Create new `UNIFIED_RESPONSE` template
5. Mark old templates inactive
6. Add tests

**Risk**: Low-Medium - Simpler than Design B due to fewer fields, but still changes core flow

---

## Comparison Matrix

| Dimension | Design A: Static Limits | Design B: LLM-Driven | Design C: Ultra-Simple LLM |
|-----------|------------------------|----------------------|---------------------------|
| **Response Quality** | ⚠️ Good (optimized per type) | ✅ Best (adaptive) | ✅ Best (adaptive) |
| **Edge Case Handling** | ❌ May miss nuances | ✅ Adapts well | ✅ Adapts well |
| **Bot Consistency** | ❌ Multiple templates | ✅ Unified voice | ✅ Unified voice |
| **Adaptability** | ❌ Static rules | ✅ Self-adapting | ✅ Self-adapting |
| **Extraction Complexity** | ⚠️ Still extracts 8 unused fields | ⚠️ Extracts 8+ fields | ✅ **Only 2 boolean flags** |
| **Extraction Reliability** | ⚠️ Complex decision (8 fields) | ⚠️ Complex decision | ✅ **Simple decision** |
| **Extraction Cost** | 200 tokens | 200 tokens | ✅ **~50 tokens (75% cheaper)** |
| **Context Quality** | ⚠️ May over/under-provision | ✅ Optimal | ✅ Optimal |
| **Predictability** | ✅ Stable context costs | ❌ Variable | ❌ Variable |
| **Implementation** | ✅ Simple (2-3 hours) | ⚠️ Moderate (4-6 hours) | ✅ **Simpler (3-4 hours)** |
| **Code Complexity** | Low | Medium | ✅ **Lowest** |
| **Risk** | ✅ Low | ⚠️ Medium | ⚠️ Low-Medium |
| **Maintenance** | ❌ Manual tuning | ✅ Self-optimizing | ✅ Self-optimizing |
| **Analytics Loss** | ✅ Keeps all fields | ✅ Keeps all fields | ⚠️ Loses unused fields |
| **Feedback Granularity** | ✅ Per template (5 types) | ⚠️ Per context pattern (4 types) | ⚠️ Per context pattern (4 types) |
| **Dreaming Evolution** | ✅ Each template evolves independently | ⚠️ Single template, context-aware | ⚠️ Single template, context-aware |

**Note**: See [Dreaming Cycle Trade-off](#critical-trade-off-dreaming-cycle-feedback-granularity) section for detailed analysis of feedback implications.

---

## Recommendation Questions

To help decide between designs, consider:

### For Design A (Static Limits):
**Best if you want**: Predictable costs and simple implementation, willing to accept multiple bot voices

1. Do you prefer **stable, predictable context costs**?
2. Are message types **clearly distinct** in your usage?
3. Is **inconsistent bot personality** (multiple templates) acceptable?
4. Are you comfortable **manually tuning** limits based on user feedback?
5. Is **simpler implementation** (2-3 hours) a priority?

If 3+ answers are "yes" → **Choose Design A**

### For Design B (LLM-Driven - Original):
**Best if you want**: Maximum quality and adaptability, willing to extract unused fields

1. Is **response quality** the top priority?
2. Do you want **unified bot personality**?
3. Is **handling edge cases** automatically important?
4. Do you need extraction analytics (entities, predicates, etc.)?
5. Are you okay with **more complex extraction** (8+ fields)?

If 3+ answers are "yes" → **Choose Design B**

### For Design C (Ultra-Simple LLM):
**Best if you want**: Maximum quality with minimal complexity, don't need analytics fields

1. Is **response quality** the top priority?
2. Do you want **unified bot personality**?
3. Is **simplicity** important (fewest fields, smallest prompt)?
4. Do you **NOT need** extraction analytics (entities, predicates, etc.)?
5. Do you want **fastest/cheapest extraction** possible?
6. Is **extraction reliability** a concern (simpler = more reliable)?

If 4+ answers are "yes" → **Choose Design C** ⭐ **Recommended for free-model usage**

---

## Recommendation for Free-Model Usage

**Given constraint: Using free LLM models (prioritize quality over tokens)**

### Design C (Ultra-Simple LLM) is the clear winner:

**Why?**
1. ✅ **Best quality** - Adaptive context like Design B
2. ✅ **Unified voice** - Single consistent template
3. ✅ **Simplest extraction** - Just 2 boolean decisions (more reliable)
4. ✅ **Fastest extraction** - 75% cheaper than current
5. ✅ **Lowest complexity** - Fewer fields = less code, easier debugging
6. ✅ **Self-adapting** - No manual tuning needed

**What you give up vs Design B:**
- ❌ Analytics fields (entities, predicates, continuation, etc.)
  - But **none of these are used for responses anyway**
  - Can always add back later if needed for analytics

**What you give up vs Design A:**
- ⚠️ Independent template evolution (5 templates → 1 template)
- ⚠️ Per-template feedback tracking (but context patterns provide similar granularity)
- **Mitigation**: Store context flags in `prompt_audits.context_config` to preserve ~80% of feedback granularity
- See [Dreaming Cycle Trade-off](#critical-trade-off-dreaming-cycle-feedback-granularity) section for details

**What you gain vs Design A:**
- ✅ Handles edge cases automatically
- ✅ Unified bot personality
- ✅ Better quality for complex queries
- ✅ No manual tuning

### Quality-First Mindset

With free models, the optimization goal shifts:
- ❌ **Don't optimize**: Token cost
- ✅ **Do optimize**: Response accuracy, consistency, adaptability

Design C achieves this by:
- Spending tokens generously when needed (complex queries get full context)
- Saving tokens when not needed (simple messages get minimal context)
- Using simplest possible extraction (most reliable)
- Maintaining single consistent voice (best UX)

---

## Next Steps

### Recommended: Implement Design C

**Implementation plan:**
1. **Create `CONTEXT_REQUEST` template** (~500 chars, just 2 boolean outputs)
2. **Simplify `QueryExtraction` dataclass** (remove 6 unused fields, keep 2 booleans)
3. **Update `bot.py`**:
   - Call new CONTEXT_REQUEST extraction
   - Conditionally retrieve based on flags:
     - Base: 10 messages, depth 0
     - +more_history: 20 messages
     - +more_triples: depth 2, max 15 triples
     - +both: 20 messages + depth 2
4. **Create `UNIFIED_RESPONSE` template** (single voice for all message types)
5. **Mark old templates inactive** (QUERY_RESPONSE, ACTIVITY_RESPONSE, etc.)
6. **Update `prompt_audits` storage** to include context flags in `context_config` JSONB
7. **Update Dreaming analyzer** to group feedback by context patterns (4 types instead of 5 templates)
8. **Test extraction reliability** (verify LLM makes good context decisions)
9. **Monitor quality metrics** (track when wrong context is requested, success rate per context pattern)

**Time estimate**: 3-4 hours

**Risk**: Low-Medium
- Simpler than Design B (fewer fields)
- Changes core flow (requires testing)
- Extraction reliability is critical (but simpler decision = more reliable)

### Fallback Strategy

If context requests prove unreliable:
- Add static caps per message type (hybrid approach)
- Or fall back to Design A with manual limits


## Summary Table

| Design | Quality | Consistency | Extraction | Implementation | Best For |
|--------|---------|-------------|------------|----------------|----------|
| **A: Static Limits** | Good | ❌ Multiple voices | Complex (8 fields) | 2-3 hours | Token optimization, predictable costs |
| **B: LLM-Driven** | ✅ Best | ✅ Unified | Complex (8+ fields) | 4-6 hours | Maximum quality, need analytics |
| **C: Ultra-Simple** | ✅ Best | ✅ Unified | ✅ **Simple (2 fields)** | 3-4 hours | ⭐ **Free models, quality-first** |

**For Lattice (2GB RAM, free models)**: **Design C** is the optimal choice.

---

## Critical Trade-off: Dreaming Cycle Feedback Granularity

### How Feedback Works Today (Design A)

The current system has a sophisticated **Dreaming Cycle** that uses user feedback to automatically optimize prompts. Here's how it works:

**1. Each response is tracked per template:**
```sql
-- prompt_audits table
prompt_key: "QUERY_RESPONSE"
template_version: 5
rendered_prompt: "..."
response_content: "..."
feedback_id: uuid (if user gave feedback)
```

**2. User feedback is template-specific:**
- User reacts with 👍/👎 to a bot message
- Feedback is linked to the specific `prompt_audit` record
- This links feedback to the exact `prompt_key` that generated the response

**3. Dreaming Cycle analyzes per template:**
```python
# From lattice/dreaming/analyzer.py
async def analyze_prompt_effectiveness():
    """Analyzes each prompt_key separately."""

    metrics_per_template = {
        "QUERY_RESPONSE": {
            "total_uses": 150,
            "positive_feedback": 12,
            "negative_feedback": 8,
            "success_rate": 0.60  # 12/(12+8)
        },
        "ACTIVITY_RESPONSE": {
            "total_uses": 300,
            "positive_feedback": 45,
            "negative_feedback": 5,
            "success_rate": 0.90  # 45/(45+5)
        },
        "CONVERSATION_RESPONSE": {
            "total_uses": 200,
            "positive_feedback": 30,
            "negative_feedback": 10,
            "success_rate": 0.75
        }
    }
```

**4. System proposes optimizations per template:**
```
If QUERY_RESPONSE success_rate < 0.70:
  → Analyze negative feedback for QUERY_RESPONSE specifically
  → Propose template changes to QUERY_RESPONSE
  → Test and measure improvement for queries
```

### The Problem with Unified Templates (Designs B & C)

**With a single UNIFIED_RESPONSE template**, all feedback goes to one place:

```sql
prompt_audits:
  prompt_key: "UNIFIED_RESPONSE"  (always)
  feedback_id: uuid

-- Loss of signal:
-- Can't tell if feedback was for:
--   - A factual query response
--   - A conversational acknowledgment
--   - An activity update response
--   - A goal planning response
```

**Impact on Dreaming Cycle:**

| Aspect | Design A (Multiple Templates) | Design B/C (Unified Template) |
|--------|------------------------------|------------------------------|
| **Feedback Granularity** | ✅ Per message type | ❌ All mixed together |
| **Problem Identification** | ✅ "QUERY_RESPONSE failing" | ❌ "Something is failing, but what?" |
| **Targeted Optimization** | ✅ "Improve query responses" | ❌ "Improve everything? Unclear." |
| **Success Measurement** | ✅ Track per template | ❌ Single aggregate metric |
| **Template Evolution** | ✅ Each evolves independently | ❌ One-size-fits-all evolution |

### Example Scenario

**Situation**: Users love conversational responses but hate query responses

**Design A (Current):**
```
Dreaming Cycle Analysis:
- CONVERSATION_RESPONSE: 90% success rate → Keep as-is
- QUERY_RESPONSE: 50% success rate → Flag for optimization
- Feedback: "Bot doesn't find my deadlines when I ask"

Action: Propose changes to QUERY_RESPONSE template specifically
```

**Design B/C (Unified):**
```
Dreaming Cycle Analysis:
- UNIFIED_RESPONSE: 75% success rate (mixed)
  - 90% of conversations → great
  - 50% of queries → poor
  - But we don't know which is which!
- Feedback: "Bot doesn't find my deadlines when I ask"

Problem: Can't isolate which message types are failing
Risk: Changing template might fix queries but break conversations
```

### Possible Mitigations for Designs B/C

**Option 1: Store message type in audit metadata**
```sql
-- Even with unified template, store extraction output
prompt_audits:
  prompt_key: "UNIFIED_RESPONSE"
  context_config: {
    "more_history": true,
    "more_triples": false,
    "inferred_type": "conversation"  -- NEW: Add back classification
  }
```

**Pros:**
- Retain feedback granularity
- Can still analyze per message type
- Dreaming Cycle can optimize differently per type

**Cons:**
- Defeats part of the simplification (still need classification)
- Template remains unified (can't evolve separately per type)
- May need to infer type from context flags

**Option 2: Feedback-based clustering**
```python
# Cluster feedback by message content patterns
negative_feedback_queries = [
    "What's my deadline?",  # User asked question
    "When did I work on X?",  # User asked question
]

negative_feedback_conversations = [
    "yeah exactly",  # User conversing
    "lol",  # User reacting
]

# Use NLP to cluster feedback by intent, not template
```

**Pros:**
- No need to store message type
- Discover patterns in feedback organically

**Cons:**
- Much more complex analysis
- Less reliable than explicit classification
- Requires significant NLP work

**Option 3: Accept coarser feedback granularity**
```
Philosophy: Unified template means unified quality bar
- If template performs poorly on queries, entire template needs improvement
- Don't optimize per-type; optimize holistically
- Use feedback text to understand what to improve
```

**Pros:**
- Simplest approach
- Aligns with "unified voice" philosophy
- Forces template to work well for ALL message types

**Cons:**
- ❌ Can't track "query responses improved 20%" metrics
- ❌ Harder to identify specific failure modes
- ❌ May over-optimize for common types, under-optimize for rare types

### Hybrid Approach: Context Flags as Pseudo-Types

**Insight**: The context request flags ARE a classification!

```json
{
  "more_history": false,
  "more_triples": false
}
→ Self-contained message (likely activity/greeting)

{
  "more_history": false,
  "more_triples": true
}
→ Factual query (needs knowledge graph)

{
  "more_history": true,
  "more_triples": false
}
→ Conversational continuation

{
  "more_history": true,
  "more_triples": true
}
→ Complex query (needs full context)
```

**Store in prompt_audits:**
```sql
prompt_audits:
  prompt_key: "UNIFIED_RESPONSE"
  context_config: {
    "more_history": true,
    "more_triples": false,
    "episodic_retrieved": 20,
    "triples_retrieved": 0
  }
```

**Dreaming analysis by context pattern:**
```sql
-- Analyze success rate by context flags
SELECT
  (context_config->>'more_history')::boolean as needs_history,
  (context_config->>'more_triples')::boolean as needs_triples,
  COUNT(*) as uses,
  AVG(CASE WHEN uf.sentiment = 'positive' THEN 1.0 ELSE 0.0 END) as success_rate
FROM prompt_audits pa
LEFT JOIN user_feedback uf ON pa.feedback_id = uf.id
WHERE pa.prompt_key = 'UNIFIED_RESPONSE'
GROUP BY needs_history, needs_triples;
```

**Result:**
```
needs_history | needs_triples | uses | success_rate
--------------+---------------+------+--------------
false         | false         | 300  | 0.90  (great - activities)
false         | true          | 150  | 0.50  (poor - queries)
true          | false         | 200  | 0.85  (good - conversation)
true          | true          | 50   | 0.60  (okay - complex)
```

**This preserves most of the granularity!**

**Dreaming Cycle can still:**
- Identify that "false/true" (queries) are failing
- Propose template improvements with note: "Focus on query responses (more_triples=true)"
- Track improvements per context pattern

### Updated Comparison

| Aspect | Design A | Design B/C + Context Tracking | Design B/C Without Tracking |
|--------|----------|------------------------------|----------------------------|
| Feedback Granularity | ✅ Excellent | ⚠️ Good (4 patterns) | ❌ Poor (1 aggregate) |
| Problem Identification | ✅ Clear | ⚠️ Requires analysis | ❌ Unclear |
| Targeted Optimization | ✅ Per template | ⚠️ Per context pattern | ❌ Holistic only |
| Template Independence | ✅ 5 templates evolve separately | ❌ 1 template for all | ❌ 1 template for all |
| Complexity | ⚠️ Higher (5 templates) | ✅ Lower (1 template) | ✅ Lower (1 template) |

### Recommendation

**For Design B or C implementation:**

1. ✅ **Store context flags in `prompt_audits.context_config`**
2. ✅ **Update Dreaming analyzer to group by context patterns**
3. ✅ **Preserve ~80% of feedback granularity** (4 patterns vs 5 templates)
4. ⚠️ **Accept loss of template independence** (can't evolve separate query/activity templates)

**Trade-off assessment:**
- Loss: Can't evolve 5 separate templates independently
- Gain: Unified voice, simpler codebase, better quality
- Mitigation: Context patterns provide sufficient granularity for optimization

**This is acceptable because:**
- Context flags map ~1:1 to message types anyway
- Single template forces holistic quality (no weak spots)
- Dreaming Cycle can still identify specific failure modes
- Benefit of unified voice outweighs loss of template independence
