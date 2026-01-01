## Executive Summary

Implement the Proactive Scheduler and Ghost Messages feature for Lattice, enabling the bot to initiate conversations based on learned patterns, user needs, and scheduled intervals. This feature allows the AI to "check in" with users proactively rather than only responding reactively.

## Background & Motivation

Lattice currently operates reactively - only responding when users message first. To become a true "companion agent," the system needs to:

1. **Proactively engage** based on learned user preferences and patterns
2. **Maintain relationship continuity** through regular check-ins
3. **Evolve its engagement strategy** based on feedback and success metrics
4. **Handle resource constraints** gracefully while maximizing user value

The proactive scheduler will be **metadata-driven**, allowing the AI to adjust its own engagement cadence through the dreaming cycle, making the proactivity behavior fully evolvable and responsive to individual user needs.

## Core Design Principles

### 1. Unified Pipeline Architecture
All behavior (Reactive User Input + Proactive Synthetic Ghosts) flows through a single unified pipeline.

### 2. Ghost Message Identity
Ghost messages are treated as **internal synthetic messages** that:
- Flow through the same processing pipeline as user messages
- Are logged to `raw_messages` with `is_bot=True`
- Can trigger short-circuit logic if they contain North Star declarations
- Are distinguished from reactive bot responses via a `ghost_origin` flag

### 3. Scheduler Metadata
The `system_health` table already contains:
- `last_proactive_eval`: Timestamp of last proactive evaluation
- `scheduled_next_proactive`: ISO timestamp for next proactive check-in

### 4. Resource Constraint Awareness
The scheduler operates within the 2GB RAM / 1vCPU constraints.

## Implementation Plan

### Phase 1: Core Scheduler Infrastructure

#### 1.1 Create Scheduler Module
**File**: `lattice/scheduler/__init__.py`, `lattice/scheduler/runner.py`

**Components**:
- `ProactiveScheduler` class that manages the scheduling loop
- Background task using `asyncio`
- Wake/sleep cycle management
- Configuration via environment variables

#### 1.2 Database Schema Updates
**Table**: `ghost_message_log` and `proactive_schedule_config`

#### 1.3 Integration with Bot Lifecycle
Extend `LatticeBot` to start/stop scheduler.

### Phase 2: Ghost Message Injection

#### 2.1 Unified Pipeline Extension
Create `lattice/core/pipeline.py` with `UnifiedPipeline` class.

#### 2.2 Ghost Context Data Structure
Create `GhostContext` and `GhostTriggerReason` in `lattice/core/types.py`.

#### 2.3 Pipeline Integration for Ghosts
Inject ghost messages through unified pipeline.

### Phase 3: Adaptive Scheduling Logic

#### 3.1 Engagement Scoring
Create `EngagementScorer` class in `lattice/scheduler/engagement.py`.

#### 3.2 Interval Adjustment
Create `AdaptiveScheduler` in `lattice/scheduler/adaptive.py`.

#### 3.3 Trigger Reason Detection
Create `TriggerDetector` in `lattice/scheduler/triggers.py`.

### Phase 4: Prompt Templates for Proactive Messages

Add PROACTIVE_CHECKIN, PROACTIVE_MILESTONE, PROACTIVE_REENGAGE templates.

### Phase 5: Ghost Message Response Handling

Extend `on_message` to handle ghost responses and update engagement metrics.

### Phase 6: Testing & Validation

Write unit tests, integration tests, and manual testing checklist.

## File Structure

```
lattice/
├── core/
│   ├── pipeline.py         # NEW: Unified pipeline for reactive + proactive
│   └── types.py            # NEW: GhostContext, GhostTriggerReason
├── scheduler/
│   ├── __init__.py
│   ├── runner.py           # NEW: ProactiveScheduler main class
│   ├── engagement.py       # NEW: EngagementScorer
│   ├── adaptive.py         # NEW: AdaptiveScheduler
│   ├── triggers.py         # NEW: TriggerDetector
│   └── response_handler.py # NEW: GhostResponseHandler
```

## Dependencies

- **No new external dependencies**
- Uses existing: `asyncpg`, `discord.py`, `structlog`, `asyncio`

## Database Schema (DDL)

### ghost_message_log Table
```sql
CREATE TABLE ghost_message_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    scheduled_for TIMESTAMPTZ NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT now(),
    trigger_reason TEXT NOT NULL CHECK (trigger_reason IN ('scheduled', 'milestone', 'engagement_lapse', 'context_switch', 'manual')),
    content_hash VARCHAR(64) CHECK (LENGTH(content_hash) = 64),  -- SHA256
    response_message_id BIGINT,                -- FK to raw_messages.discord_message_id
    response_within_minutes INT,               -- NULL if no response
    user_reaction_count INT DEFAULT 0 CHECK (user_reaction_count >= 0),
    user_reply_count INT DEFAULT 0 CHECK (user_reply_count >= 0),
    engagement_score FLOAT CHECK (engagement_score >= 0 AND engagement_score <= 1),
    was_appropriate BOOLEAN,                   -- NULL until feedback received
    context_used JSONB CHECK (jsonb_typeof(context_used) = 'object'),
    response_preview TEXT CHECK (LENGTH(response_preview) <= 200),
    metadata JSONB DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_ghost_log_channel_time ON ghost_message_log(channel_id, triggered_at DESC);
CREATE INDEX idx_ghost_log_user ON ghost_message_log(user_id, triggered_at DESC);
CREATE INDEX idx_ghost_log_hash ON ghost_message_log(content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX idx_ghost_log_success ON ghost_message_log(was_appropriate) WHERE was_appropriate = true;
CREATE INDEX idx_ghost_log_triggered_at ON ghost_message_log(triggered_at DESC);
CREATE INDEX idx_ghost_log_reason ON ghost_message_log(trigger_reason, triggered_at DESC);
```

### proactive_schedule_config Table
```sql
CREATE TABLE proactive_schedule_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL UNIQUE,
    channel_id BIGINT NOT NULL,
    base_interval_minutes INT NOT NULL DEFAULT 60 CHECK (base_interval_minutes >= 15 AND base_interval_minutes <= 10080),
    current_interval_minutes INT NOT NULL DEFAULT 60 CHECK (current_interval_minutes >= 15 AND current_interval_minutes <= 10080),
    min_interval_minutes INT NOT NULL DEFAULT 15 CHECK (min_interval_minutes >= 5 AND min_interval_minutes <= 10080),
    max_interval_minutes INT NOT NULL DEFAULT 10080 CHECK (max_interval_minutes <= 604800),
    CHECK (current_interval_minutes >= min_interval_minutes),
    CHECK (current_interval_minutes <= max_interval_minutes),
    last_engagement_score FLOAT,
    engagement_count INT DEFAULT 0,
    is_paused BOOLEAN DEFAULT false,
    paused_reason TEXT,
    opt_out BOOLEAN DEFAULT false,             -- User can opt out of proactive
    opt_out_at TIMESTAMPTZ,                    -- When user opted out
    user_timezone TEXT DEFAULT 'UTC',          -- For scheduling at local times
    active_triggers TEXT[] DEFAULT ARRAY['scheduled']::TEXT[],
    last_ghost_at TIMESTAMPTZ,
    next_scheduled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_schedule_config_user ON proactive_schedule_config(user_id);
CREATE INDEX idx_schedule_config_opt_out ON proactive_schedule_config(opt_out) WHERE opt_out = false;
CREATE INDEX idx_schedule_config_due ON proactive_schedule_config(next_scheduled_at)
    WHERE is_paused = false AND opt_out = false;
CREATE INDEX idx_schedule_config_paused ON proactive_schedule_config(is_paused, opt_out, next_scheduled_at);
```

### Trigger Enum (Application-Level)
```python
from enum import Enum

class GhostTriggerReason(Enum):
    SCHEDULED = "scheduled"           # Routine check-in based on interval
    MILESTONE = "milestone"           # User achieved a tracked goal
    ENGAGEMENT_LAPSE = "engagement_lapse"  # User inactive for extended period
    CONTEXT_SWITCH = "context_switch"  # New topic introduced elsewhere
    MANUAL = "manual"                  # User explicitly requested check-in
```

### Pipeline Source Type (Short-Circuit Safety)
```python
from enum import Enum

class PipelineSourceType(Enum):
    USER = "user"         # Reactive message from user
    GHOST = "ghost"       # Proactive synthetic message
    SYSTEM = "system"     # Internal system signal
```

## Safety & User Autonomy

### Content Safety Filter
```python
class GhostContentSafetyFilter:
    async def filter(self, content: str) -> tuple[bool, str]:
        """Filter ghost content before sending.
        
        Returns:
            (is_safe, reason_if_not_safe)
        """
        # 1. Length check
        if len(content) > 1900:
            return False, "Content too long"
        
        # 2. Profanity/harmful content check
        if await self._contains_harmful_content(content):
            return False, "Contains harmful content"
        
        # 3. Self-reference check
        if "I am a bot" in content or "I was programmed" in content:
            return False, "Contains self-reference"
        
        # 4. Hallucination check - verify against known facts
        if await self._contains_unverified_claims(content):
            return False, "Contains unverified claims"
        
        return True, ""
```

### User Opt-Out Mechanism
```python
class UserOptOutManager:
    async def handle_opt_out(self, user_id: int) -> None:
        """Handle user requesting opt-out from proactive messages."""
        async with db_pool.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE proactive_schedule_config
                SET opt_out = true, updated_at = now()
                WHERE user_id = $1
                """,
                user_id,
            )
    
    async def is_opted_out(self, user_id: int) -> bool:
        """Check if user has opted out."""
        async with db_pool.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT opt_out FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )
```

### Recursion Guard
```python
class RecursionGuard:
    MAX_GHOST_DEPTH = 2  # Max ghosts that can chain
    
    def __init__(self) -> None:
        self._depth: int = 0
    
    async def enter_ghost(self) -> bool:
        """Enter ghost context. Returns False if max depth exceeded."""
        if self._depth >= self.MAX_GHOST_DEPTH:
            logger.warning("Max ghost recursion depth exceeded")
            return False
        self._depth += 1
        return True
    
    def exit_ghost(self) -> None:
        """Exit ghost context."""
        self._depth = max(0, self._depth - 1)
```

## Engagement Scoring Algorithm

```python
import random
from datetime import timedelta

class EngagementScorer:
    WEIGHTS = {
        "response_rate": 0.35,        # User responds to ghosts
        "message_frequency": 0.25,     # User sends messages overall
        "feedback_positive": 0.25,     # Positive 🫡 feedback
        "conversation_depth": 0.15,    # Response length/quality
    }
    
    async def calculate_score(self, user_id: int, window_hours: int = 24) -> float:
        """Calculate engagement score for a user.
        
        Returns:
            Score between 0.0 (low engagement) and 1.0 (high engagement)
        """
        async with db_pool.pool.acquire() as conn:
            # Get user's channel_id first (avoid repeating subquery)
            channel_id = await conn.fetchval(
                "SELECT channel_id FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )
            if not channel_id:
                return 0.5  # Default for new users
            
            # Get metrics from last N hours
            metrics = await conn.fetchrow("""
                SELECT
                    -- Response rate: responses / ghost_messages
                    COALESCE(
                        SUM(CASE WHEN response_message_id IS NOT NULL THEN 1 ELSE 0 END)::FLOAT /
                        NULLIF(SUM(1), 0),
                        0.5
                    ) AS response_rate,
                    
                    -- Message frequency: messages per hour
                    COALESCE(
                        (SELECT COUNT(*) FROM raw_messages
                         WHERE channel_id = $1
                         AND timestamp > now() - interval '%s hours')::FLOAT / $2,
                        0.0
                    ) AS message_frequency,
                    
                    -- Feedback positive: 🫡 reactions / total feedback
                    COALESCE(
                        (SELECT COUNT(*) FROM user_feedback WHERE reaction = '🫡')::FLOAT /
                        NULLIF((SELECT COUNT(*) FROM user_feedback), 0),
                        0.5
                    ) AS feedback_positive,
                    
                    -- Conversation depth: avg response length
                    COALESCE(
                        (SELECT AVG(LENGTH(content)) FROM raw_messages
                         WHERE is_bot = false
                         AND channel_id = $1
                         AND timestamp > now() - interval '%s hours')::FLOAT / 500.0,
                        0.5
                    ) AS conversation_depth
                FROM ghost_message_log
                WHERE user_id = $2
                AND triggered_at > now() - interval '%s hours'
            """, channel_id, f"{window_hours}", window_hours, f"{window_hours}", user_id, f"{window_hours}")
        
        # Normalize and weight
        score = (
            metrics["response_rate"] * self.WEIGHTS["response_rate"] +
            min(metrics["message_frequency"], 1.0) * self.WEIGHTS["message_frequency"] +
            metrics["feedback_positive"] * self.WEIGHTS["feedback_positive"] +
            min(metrics["conversation_depth"], 1.0) * self.WEIGHTS["conversation_depth"]
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
```

**Note**: The SQL uses string formatting for interval values to work around asyncpg parameter limitations. In production, use parameterized queries or raw SQL files.

## Scheduler Loop Implementation

```python
import random
import pytz
from datetime import timedelta
from discord import Permissions

class ProactiveScheduler:
    MAX_CONSECUTIVE_FAILURES = 5
    FAILURE_BACKOFF_SECONDS = 300  # 5 minutes
    
    def __init__(
        self,
        bot: LatticeBot,
        check_interval: int = 60,
        initial_delay: int = 300,
    ) -> None:
        self.bot = bot
        self.check_interval = check_interval
        self.initial_delay = initial_delay
        self._running = False
        self._recursion_guard = RecursionGuard()
        self._consecutive_failures = 0
        self._last_failure_time: datetime | None = None
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Starting proactive scheduler", check_interval=self.check_interval)
        
        # Initial delay before first proactive
        await asyncio.sleep(self.initial_delay)
        
        asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False
        logger.info("Stopping proactive scheduler")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Circuit breaker - back off after consecutive failures
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    backoff_time = min(
                        self.FAILURE_BACKOFF_SECONDS * (2 ** (self._consecutive_failures - self.MAX_CONSECUTIVE_FAILURES)),
                        3600  # Max 1 hour backoff
                    )
                    logger.warning(
                        "Scheduler circuit breaker activated",
                        consecutive_failures=self._consecutive_failures,
                        backoff_seconds=backoff_time,
                    )
                    await asyncio.sleep(backoff_time)
                    self._consecutive_failures = 0  # Reset on successful cycle
                
                now = datetime.now(UTC)
                
                # Get due ghosts using atomic check with skip locked
                due_ghosts = await self._get_due_ghosts(now)
                
                for ghost_config in due_ghosts:
                    if not self._running:
                        break
                    
                    # Check opt-out and pause status
                    if ghost_config.opt_out or ghost_config.is_paused:
                        continue
                    
                    # Check timezone and local time
                    if not self._is_local_time_active(ghost_config.user_timezone):
                        continue
                    
                    # Check Discord permissions before attempting send
                    if not await self._has_send_permission(ghost_config.channel_id):
                        logger.warning("Missing send permission", channel_id=ghost_config.channel_id)
                        continue
                    
                    # Apply jitter (±10%)
                    jitter = random.uniform(-0.1, 0.1)
                    next_interval = int(ghost_config.current_interval_minutes * (1 + jitter))
                    
                    # Execute ghost
                    success = await self._execute_ghost(ghost_config)
                    
                    if success:
                        self._consecutive_failures = 0  # Reset on success
                        # Schedule next
                        next_at = now + timedelta(minutes=next_interval)
                        await self._update_next_scheduled(ghost_config.user_id, next_at)
                    else:
                        self._consecutive_failures += 1
                        self._last_failure_time = now
                
            except Exception as e:
                self._consecutive_failures += 1
                logger.exception("Scheduler loop error", error=str(e))
            
            await asyncio.sleep(self.check_interval)
    
    async def _get_due_ghosts(self, now: datetime) -> list[ProactiveScheduleConfig]:
        """Get all ghosts scheduled for before now with skip-locked for concurrency."""
        async with db_pool.pool.acquire() as conn:
            return await conn.fetch("""
                SELECT * FROM proactive_schedule_config
                WHERE opt_out = false
                AND is_paused = false
                AND (next_scheduled_at IS NULL OR next_scheduled_at <= $1)
                ORDER BY next_scheduled_at ASC NULLS FIRST
                LIMIT 50
                FOR UPDATE SKIP LOCKED
            """, now)
    
    async def _is_local_time_active(self, timezone_str: str) -> bool:
        """Check if current local time is within active hours (9 AM - 9 PM)."""
        try:
            tz = pytz.timezone(timezone_str)
            local_now = datetime.now(tz)
            local_hour = local_now.hour
            return 9 <= local_hour <= 21  # Active hours: 9 AM to 9 PM
        except pytz.UnknownTimeZoneError:
            return True  # Default to active if unknown timezone
    
    async def _has_send_permission(self, channel_id: int) -> bool:
        """Check if bot has permission to send messages in channel."""
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return False
        
        permissions = channel.permissions_for(channel.guild.me)
        return permissions.send_messages
    
    async def _execute_ghost(self, config: ProactiveScheduleConfig) -> bool:
        """Execute a single ghost message."""
        # Recursion guard
        can_proceed = await self._recursion_guard.enter_ghost()
        if not can_proceed:
            return False
        
        try:
            # Build ghost context
            context = await self._build_ghost_context(config)
            
            # Generate content
            content = await self._generate_ghost_content(context)
            
            # Safety filter
            is_safe, reason = await self._safety_filter.filter(content)
            if not is_safe:
                logger.warning("Ghost content filtered", user_id=config.user_id, reason=reason)
                return False
            
            # Send via unified pipeline
            result = await self._unified_pipeline.process_message(
                content=content,
                channel_id=config.channel_id,
                is_ghost=True,
                ghost_context=context,
            )
            
            # Log success
            await self._log_ghost_execution(config, context, result)
            return True
            
        except discord.RateLimited as e:
            logger.warning("Rate limited during ghost send", retry_after=e.retry_after)
            return False
        except discord.Forbidden:
            logger.warning("Forbidden - missing permissions", channel_id=config.channel_id)
            return False
        finally:
            self._recursion_guard.exit_ghost()
```

## Rollout Strategy

1. **Disabled by Default**: Feature requires `SCHEDULER_ENABLED=true`
2. **Gradual Ramp-up**: Start with high intervals, decrease based on engagement
3. **Monitoring**: Log all ghost messages and engagement metrics
4. **Feedback Loop**: Use invisible feedback (🫡) to adjust behavior
5. **User Opt-Out**: Users can disable via `opt_out` flag

## Rollout Strategy

1. **Disabled by Default**: Feature requires `SCHEDULER_ENABLED=true`
2. **Gradual Ramp-up**: Start with high intervals, decrease based on engagement
3. **Monitoring**: Log all ghost messages and engagement metrics
4. **Feedback Loop**: Use invisible feedback (🫡) to adjust behavior

## Implementation Tasks

### Task 1: Scheduler Infrastructure
- [ ] Create `lattice/scheduler/__init__.py`
- [ ] Create `lattice/scheduler/runner.py` with ProactiveScheduler class
- [ ] Add environment variable configuration
- [ ] Integrate scheduler start/stop with bot lifecycle
- [ ] Write unit tests for scheduler

### Task 2: Database Schema
- [ ] Create migration for `ghost_message_log` table
- [ ] Create migration for `proactive_schedule_config` table
- [ ] Write tests for DB operations

### Task 3: Unified Pipeline
- [ ] Create `lattice/core/pipeline.py` with UnifiedPipeline class
- [ ] Refactor bot's `on_message` to use unified pipeline
- [ ] Add ghost-specific handling in pipeline
- [ ] Write integration tests

### Task 4: Ghost Message Generation
- [ ] Add PROACTIVE_* prompt templates to registry
- [ ] Create `lattice/core/types.py` with GhostContext
- [ ] Implement ghost content generation logic
- [ ] Write tests for ghost generation

### Task 5: Engagement & Adaptation
- [ ] Create `lattice/scheduler/engagement.py` with EngagementScorer
- [ ] Create `lattice/scheduler/adaptive.py` with AdaptiveScheduler
- [ ] Implement interval adjustment logic
- [ ] Write tests for engagement scoring

### Task 6: Trigger Detection
- [ ] Create `lattice/scheduler/triggers.py` with TriggerDetector
- [ ] Implement milestone detection
- [ ] Implement engagement lapse detection
- [ ] Write tests for triggers

### Task 7: Response Handling
- [ ] Extend `on_message` to detect ghost responses
- [ ] Create `lattice/scheduler/response_handler.py`
- [ ] Implement engagement metric updates
- [ ] Write integration tests

### Task 8: End-to-End Testing
- [ ] Run full scheduler flow tests
- [ ] Test with real Discord integration (manual)
- [ ] Performance testing under load
- [ ] Document troubleshooting guide

---

## Reviewer Feedback (Iteration 1)

### Architecture and Design Feedback

**Strengths:**
- The unified pipeline concept is sound and aligns with the existing ENGRAM philosophy of metadata-driven behavior. Treating ghost messages as "internal synthetic messages" through the same pipeline ensures consistency and reduces code duplication.

**Concerns:**
1. **Short-circuit logic collision risk**: If ghost messages can trigger short-circuit logic (e.g., North Star declarations), there's potential for recursive or infinite loops. A ghost message containing a North Star declaration could trigger another ghost message.
2. **Pipeline state contamination**: Ghost messages originate from the system, not users. Does the pipeline need to differentiate between "user context" and "system context"?
3. **Missing: Rate limiting at pipeline level**: The proposal doesn't address how rate limiting interacts with the unified pipeline.

**Recommendations:**
- Add `source_type` enum to pipeline context (`USER`, `GHOST`, `SYSTEM`)
- Implement recursion depth tracking in pipeline
- Add explicit "ghost cannot trigger ghost" short-circuit rule

### Database Schema Feedback

**Missing schema details:**
The issue mentions tables but doesn't provide column definitions, data types, constraints, or indexes.

**Required schema elements for `ghost_message_log`:**
```
- id: UUID PRIMARY KEY
- created_at: TIMESTAMPTZ DEFAULT NOW()
- trigger_reason: ENUM (MILESTONE, ENGAGEMENT_LAPSE, TIMEOUT, etc.)
- content_hash: VARCHAR(64) -- deduplication
- engagement_score: FLOAT -- at time of generation
- user_response_id: UUID FK to raw_messages (nullable)
- was_appropriate: BOOLEAN NULL -- from feedback
- metadata: JSONB
```

**Required schema elements for `proactive_schedule_config`:**
```
- user_id: UUID PRIMARY KEY FK
- base_interval_minutes: INT
- current_interval_minutes: INT
- min_interval_minutes: INT CHECK >= 15
- max_interval_minutes: INT CHECK <= 10080
- engagement_threshold: FLOAT
- last_adjustment_at: TIMESTAMPTZ
- active_triggers: TEXT[]
```

**Concern:** No mention of `ghost_message_log` cleanup/archival policy.

### Scheduler Logic Feedback

**Concerns:**
1. **Concurrency safety**: What happens if the bot restarts between `scheduled_next_proactive` evaluation and message generation?
2. **Engagement scoring ambiguity**: The formula and inputs aren't specified.
3. **Adaptive bounds missing**: The issue doesn't specify minimum/maximum interval bounds.
4. **No jitter/randomization**: Deterministic scheduling is problematic.
5. **Timezone handling**: Users are in different timezones.

**Recommendations:**
- Add deterministic jitter (±10% of interval)
- Hard-code MIN_INTERVAL (15 min) and MAX_INTERVAL (7 days) in schema CHECK constraints
- Add `user_timezone` field to user preferences
- Use atomic compare-and-swap for `scheduled_next_proactive`

### Testing Strategy Feedback

**Missing test cases:**
1. Scheduler isolation tests
2. Ghost message lifecycle tests
3. Adaptive scheduling tests
4. Edge cases (user blocks/mutes bot, rate limit hit, database unavailable)
5. Performance tests (1000+ users, memory impact)

### Missing Components

1. **User Preferences Table**: No place to store per-user schedule preferences
2. **Graceful Degradation Path**: What happens when LLM fails, embedding service unavailable, DB exhausted?
3. **Ghost Content Safety Filter**: Need content filtering before sending
4. **Channel/Guild Awareness**: Need channel permission checks
5. **Ghost Response Detection**: How to distinguish ghost response from new conversation?

### Security and Safety

**Content Safety:**
- Ghost messages are auto-generated without human review
- No content moderation pipeline described
- Risk of hallucinated facts

**User Autonomy:**
- No opt-out mechanism described
- No "snooze" functionality for temporary quiet periods

### Overall Assessment

The proposal is a solid foundation. However, the implementation lacks critical detail in several areas, particularly:
- Database schema column definitions
- Safety mechanisms (content filtering, user opt-out, recursion guards)
- Specific algorithm details for engagement scoring and interval adaptation

**Recommended before implementation:**
1. Add detailed schema DDL with constraints
2. Define the EngagementScorer formula explicitly
3. Add safety section with opt-out mechanism
4. Document the `GhostTriggerReason` enum values
5. Create explicit failure mode documentation

---

## Reviewer Feedback (Iteration 2)

### What Was Addressed

The iteration 1 feedback was **acknowledged** but not **implemented** in the spec. The feedback section documents concerns but doesn't show updated spec text addressing them.

### What Remains

**Critical Gaps:**

1. **No DDL for tables**: `ghost_message_log` and `proactive_schedule_config` are referenced but no CREATE TABLE statements exist
2. **EngagementScorer formula undefined**: No mathematical specification
3. **`GhostTriggerReason` enum values undefined**: What triggers are actually supported?
4. **Short-circuit collision resolution**: Still only listed as a concern, no `source_type` enum or recursion guard
5. **No opt-out mechanism**: User autonomy concerns unaddressed
6. **No content safety filter**: Ghost messages could hallucinate or send inappropriate content

### New Issues

- No schema CHECK constraints documented for MIN/MAX interval bounds
- Missing `user_timezone` handling - scheduler will fire at wrong times
- No atomic operations specified for `scheduled_next_proactive` updates

### Implementation Clarity

**Not implementation-ready.** A developer cannot build from this spec because:
1. Database schema requires reverse-engineering from vague descriptions
2. Engagement scoring algorithm is unspecified
3. No explicit pseudocode for the scheduler loop
4. No failure mode documentation

### Priority Items Before Implementation

**Must-have (blocking):**

1. Add DDL for `ghost_message_log` and `proactive_schedule_config` with all columns, types, constraints, and indexes
2. Define `GhostTriggerReason` enum with all supported values
3. Specify engagement scoring formula with concrete weights and inputs
4. Add safety section with:
   - User opt-out mechanism
   - Content moderation before send
   - Recursion depth guard
   - `source_type` enum (`USER`, `GHOST`, `SYSTEM`)
5. Add CHECK constraints for interval bounds (MIN 15 min, MAX 7 days)
6. Add `user_timezone` field to user preferences

### Code Examples Needed

**GhostContext (missing concrete structure):**
```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

class GhostTriggerReason(Enum):
    MILESTONE = "milestone"
    ENGAGEMENT_LAPSE = "engagement_lapse"
    TIMEOUT = "timeout"
    SCHEDULED = "scheduled"

@dataclass
class GhostContext:
    user_id: UUID
    trigger_reason: GhostTriggerReason
    engagement_score: float
    recent_context: list[dict]
    last_ghost_at: datetime | None
    content_hash: str | None  # Deduplication
```

**EngagementScorer (undefined):**
```python
class EngagementScorer:
    async def calculate_score(self, user_id: UUID) -> float:
        # What are the inputs?
        # msg_frequency_weight = ?
        # response_rate_weight = ?
        # feedback_positive_weight = ?
        raise NotImplementedError
```

### Summary

The issue is a good **architecture document** but not an **implementation spec**.

---

## Reviewer Feedback (Iteration 3)

### Overall Assessment

**Status: CONDITIONALLY APPROVED for implementation**

The issue has improved significantly. However, several issues remain that should be fixed before implementation.

### Code Review Findings

**Type Safety Issues:**
- `GhostContext` import is missing in the code snippet
- `EngagementScorer.calculate_score()` has a **SQL bug** - 4 placeholders but only 3 parameters

**Error Handling Issues:**
- No circuit breaker pattern in scheduler loop
- No handling for `discord.RateLimited`, `discord.Forbidden`
- `_contains_harmful_content()` not implemented

**Concurrency Safety Issues:**
- `RecursionGuard` uses `self._depth` without thread safety (should use `asyncio.Lock`)
- Race condition in `_get_due_ghosts()` - needs `FOR UPDATE SKIP LOCKED`
- No atomic compare-and-swap for `next_scheduled_at`

**Resource Management Concerns:**
- `ghost_message_log.metadata JSONB` - no size limit specified
- Inefficient subquery in EngagementScorer (runs 4 times)

### Schema Review Findings

**Constraint Issues:**
- `current_interval_minutes` constraint conflicts with `min/max_interval_minutes` columns
- Need additional constraint: `CHECK (current_interval_minutes >= min_interval_minutes)`

**Index Issues:**
- Missing index on `opt_out` for filtering
- Missing composite index for due ghosts query
- Missing index on `ghost_message_log.triggered_at`

### Critical Bugs to Fix

1. **SQL parameter count bug** in `EngagementScorer.calculate_score()`:
```python
# WRONG:
""", user_id, window_hours, window_hours, window_hours)  # 4 placeholders, 3 params

# CORRECT:
""", user_id, window_hours, window_hours, window_hours, window_hours)
```

2. **Missing `random` import** in scheduler loop

3. **Timezone handling missing** - scheduler uses `datetime.now(UTC)` without conversion

4. **Missing Discord permission checks** before sending ghost messages

### Final Recommendation

**APPROVE FOR IMPLEMENTATION with required fixes:**

**Must Fix Before Implementation:**
1. Fix SQL parameter count bug
2. Add missing CHECK constraints for interval bounds
3. Add `random` import
4. Add `FOR UPDATE SKIP LOCKED` to `_get_due_ghosts()` query
5. Implement timezone conversion
6. Add Discord permission check before sending
7. Implement atomic update for `next_scheduled_at`
8. Implement or document content moderation

**Should Add Before Implementation:**
1. Circuit breaker in scheduler loop
2. Composite index for filtering
3. Index on `triggered_at`
4. Metrics logging for scheduler health