# Concurrency Model & Consolidation Analysis

## Bot Concurrency Model

### Current Implementation

**Discord.py Event Loop** (`lattice/discord_client/bot.py:148`)

```python
async def on_message(self, message: discord.Message) -> None:
    """Handle incoming Discord messages."""
    # ... validation and routing ...

    # Store, extract, retrieve, generate, send
    user_message_id = await memory_orchestrator.store_user_message(...)
    extraction = await query_extraction.extract_query_structure(...)
    recent_messages, graph_triples = await memory_orchestrator.retrieve_context(...)
    response_result, rendered_prompt, context_info = await response_generator.generate_response(...)

    # Send response
    for msg in response_messages:
        bot_msg = await message.channel.send(msg)

    # Fire-and-forget consolidation
    await memory_orchestrator.consolidate_message_async(...)  # Line 370
```

### How Discord.py Handles Concurrency

**Single Event Loop, Multiple Concurrent Handlers**

Discord.py uses **asyncio** with a single event loop but processes events concurrently:

1. **Each `on_message` call runs independently** in the event loop
2. **Multiple messages can be processed simultaneously** (async concurrency)
3. **Not truly parallel** (single Python thread), but concurrent via asyncio
4. **No explicit queue** - Discord.py's internal event dispatcher handles ordering

**What happens if message arrives during processing?**

```
Message 1: "Hello" arrives
  ↓ on_message(msg1) starts
    ↓ await query_extraction (yielding control)

Message 2: "How are you?" arrives
  ↓ on_message(msg2) starts (CONCURRENT with msg1!)
    ↓ await query_extraction (yielding control)

Message 1: Query extraction completes
  ↓ continues to response generation

Message 2: Query extraction completes  
  ↓ continues to response generation

Both messages processed concurrently, not sequentially!
```

### Current Behavior

✅ **Bot CAN process multiple messages concurrently**
- Discord.py automatically dispatches events
- Async functions yield control at `await` points
- Multiple handlers can be in-flight simultaneously

⚠️ **Potential race conditions**:
- Database writes (mitigated by PostgreSQL ACID guarantees)
- Memory state mutations (currently minimal)
- Consolidation overlap (fire-and-forget, no coordination)

## Consolidation Timing Analysis

### Why Consolidate AFTER Response?

**Current flow** (`bot.py:210-378`):

```python
# 1. Store user message
user_message_id = await store_user_message(...)

# 2. Extract query structure
extraction = await extract_query_structure(...)

# 3. Retrieve context
recent_messages, graph_triples = await retrieve_context(...)

# 4. Generate response
response_result = await generate_response(...)

# 5. Send response to Discord
bot_msg = await message.channel.send(msg)

# 6. Consolidate (AFTER send)
await consolidate_message_async(...)  # Line 370
```

### Why Not Earlier?

#### Option 1: Consolidate BEFORE Sending Response

```python
# Generate response
response_result = await generate_response(...)

# Consolidate FIRST (blocks response)
await consolidate_message(...)  # Blocks ~500ms-1s

# THEN send response (delayed!)
bot_msg = await message.channel.send(msg)
```

**Problems**:
- ❌ **Higher perceived latency** - User waits for consolidation
- ❌ **Consolidation doesn't help current response** - Extracted facts not used yet
- ❌ **LLM cost on critical path** - TRIPLE_EXTRACTION call delays response

#### Option 2: Consolidate in Parallel with Response

```python
# Generate response
response_result = await generate_response(...)

# Send response AND consolidate in parallel
await asyncio.gather(
    message.channel.send(msg),
    consolidate_message(...)
)
```

**Problems**:
- ⚠️ **Consolidation still delays send** - `gather()` waits for slowest task
- ⚠️ **No benefit** - Still blocking on consolidation

### Why Async Fire-and-Forget?

**Current implementation** (`memory_orchestrator.py:164-204`):

```python
async def consolidate_message_async(...) -> None:
    """Start background consolidation (fire-and-forget).

    This creates a background task without blocking the caller.
    Errors in consolidation are logged but not propagated.
    """
    _consolidation_task = asyncio.create_task(  # noqa: RUF006
        episodic.consolidate_message(...)
    )
```

**Reasoning**:
1. ✅ **Response sent immediately** - No blocking on extraction
2. ✅ **Consolidation happens in background** - Async task continues independently
3. ✅ **Bot stays responsive** - Can process new messages while consolidating
4. ✅ **Errors isolated** - Consolidation failure doesn't break response flow

### Why Not Synchronous?

If consolidation were synchronous:

```python
# Send response
bot_msg = await message.channel.send(msg)

# Block on consolidation (BAD!)
await consolidate_message(...)  # Blocks ~500ms-1s

# Can't process new messages until done!
```

**Problems**:
- ❌ **Bot appears slow/laggy** - Busy consolidating instead of responding
- ❌ **Throughput limited** - One message at a time
- ❌ **Bad user experience** - Delay between messages

## Race Condition Analysis

### Potential Issues

#### 1. Overlapping Consolidation

**Scenario**: Two messages arrive quickly

```
Message 1: "My mom's birthday is March 15"
  ↓ Consolidation starts in background
    ↓ Extracting entities...

Message 2: "Her favorite color is blue"
  ↓ Consolidation starts in background (OVERLAPS!)
    ↓ Extracting entities...

Both tasks try to INSERT entity "mom" simultaneously
```

**Mitigation**: PostgreSQL handles this with `ON CONFLICT DO NOTHING` (if implemented)

#### 2. Context Retrieval Before Consolidation Complete

**Scenario**: Fast typer or multiple users

```
Message 1: "Remember my birthday is June 1st"
  ↓ Response sent immediately
  ↓ Consolidation starts extracting...

Message 2: "When is my birthday?" (arrives 200ms later)
  ↓ Context retrieval runs
  ↓ Consolidation from Message 1 NOT done yet!
  ↓ Birthday fact missing from graph!
  ↓ Response: "I don't have that information"
```

**This is acceptable** because:
- ✅ **Episodic memory always available** - Message 1 content still in recent messages
- ✅ **Eventual consistency** - Future queries will find the fact
- ✅ **Tradeoff for responsiveness** - Better than blocking

### Should We Add a Queue?

**Proposed queue system**:

```python
class ConsolidationQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.worker_task = None

    async def start_worker(self):
        """Single worker processes consolidations sequentially."""
        while True:
            message_id, content, context = await self.queue.get()
            await consolidate_message(message_id, content, context)
            self.queue.task_done()

    async def enqueue(self, message_id, content, context):
        """Add to queue instead of spawning task."""
        await self.queue.put((message_id, content, context))
```

**Analysis**:

✅ **Pros**:
- Prevents overlapping consolidation
- Sequential processing (predictable)
- Backpressure handling (queue size limits)

❌ **Cons**:
- **Slower consolidation** - No parallel processing
- **Added complexity** - Queue management, monitoring
- **Not necessary yet** - Current race conditions manageable

**Recommendation**: **Do NOT add queue for now**
- PostgreSQL handles concurrent writes safely
- Consolidation is idempotent (safe to retry)
- No evidence of actual problems from races
- Can add later if issues arise

## Re-enabling Triple Extraction

### Current State

**Triple extraction is disabled** (`episodic.py:237-243`):

```python
triples = parse_triples(result.content)
logger.info("Parsed triples", count=len(triples) if triples else 0)

# TODO: Consolidation is temporarily disabled during Issue #61 refactor
# Will be reimplemented with query extraction + graph-first architecture
logger.debug(
    "Skipping triple consolidation (disabled during Issue #61 refactor)",
    message_id=str(message_id),
    triple_count=len(triples) if triples else 0,
)
```

### Issue #61 Status

**Issue #61** was about implementing graph-first architecture with query extraction.

**Status**: ✅ **CLOSED** (merged in #87)

Key deliverables from #61:
- ✅ Query extraction layer (2-field simplified version in #87)
- ✅ Graph-first schema (entities + semantic_triples tables exist)
- ✅ Entity-driven context retrieval (Design D in #87)
- ❌ **Triple storage NOT re-enabled** (commented out)

### Re-enabling Triple Storage

**Required changes**:

1. **Uncomment storage logic** (`episodic.py:237-243`):

```python
triples = parse_triples(result.content)
logger.info("Parsed triples", count=len(triples) if triples else 0)

# Re-enable triple storage (Issue #61 complete, #87 merged)
if triples:
    await store_semantic_triples(
        message_id=message_id,
        triples=triples
    )
    logger.info(
        "Stored semantic triples",
        message_id=str(message_id),
        count=len(triples)
    )
```

2. **Implement `store_semantic_triples()`** (new function):

```python
async def store_semantic_triples(
    message_id: UUID,
    triples: list[dict[str, str]]
) -> None:
    """Store extracted triples in semantic_triples table.

    Args:
        message_id: UUID of origin message (for origin_id FK)
        triples: List of {"subject": str, "predicate": str, "object": str}
    """
    async with db_pool.pool.acquire() as conn, conn.transaction():
        for triple in triples:
            # 1. Upsert entities (subject and object)
            subject_id = await upsert_entity(
                conn,
                name=triple["subject"]
            )
            object_id = await upsert_entity(
                conn,
                name=triple["object"]
            )

            # 2. Insert triple with origin_id link
            await conn.execute(
                """
                INSERT INTO semantic_triples (
                    subject_id, predicate, object_id, origin_id
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                subject_id,
                triple["predicate"],
                object_id,
                message_id  # Links to raw_messages(id)
            )


async def upsert_entity(
    conn: asyncpg.Connection,
    name: str,
    entity_type: str | None = None
) -> UUID:
    """Get or create entity by name.

    Args:
        conn: Database connection
        name: Entity name (normalized)
        entity_type: Optional entity type

    Returns:
        UUID of entity
    """
    # Normalize entity name
    normalized_name = name.lower().strip()

    # Try to find existing
    row = await conn.fetchrow(
        "SELECT id FROM entities WHERE LOWER(name) = $1",
        normalized_name
    )

    if row:
        return row["id"]

    # Create new entity
    row = await conn.fetchrow(
        """
        INSERT INTO entities (name, entity_type)
        VALUES ($1, $2)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        name,  # Use original casing
        entity_type
    )

    return row["id"]
```

3. **Add entity resolution** (optional, see #88):

Entity resolution for synonyms (e.g., "mom" vs "mother") can be added later as enhancement.

### Testing Re-enabled Triple Storage

```python
async def test_triple_extraction_and_storage():
    """Test end-to-end triple extraction and storage."""
    # 1. Send message
    message = "Alice works at OpenAI"
    message_id = await store_user_message(content=message, ...)

    # 2. Consolidate (extracts and stores triples)
    await consolidate_message(
        message_id=message_id,
        content=message,
        context=[]
    )

    # 3. Verify entities created
    entities = await get_entities()
    assert "alice" in [e["name"].lower() for e in entities]
    assert "openai" in [e["name"].lower() for e in entities]

    # 4. Verify triple stored
    triples = await get_semantic_triples()
    assert len(triples) == 1
    assert triples[0]["predicate"] == "works_at"
    assert triples[0]["origin_id"] == message_id  # Links to source!
```

## Link Injection Strategy

### Requirements

From user:
> "i dont want it to be intrusive, just link icons for each source is sufficient"

### Proposed Implementation

**Non-intrusive footnote links** with emoji icons:

```python
def inject_source_links(
    response: str,
    sources: list[tuple[str, str]]
) -> str:
    """Inject minimal source links at end of response.

    Args:
        response: Generated response text
        sources: [(preview_text, jump_url), ...]

    Returns:
        Response with subtle source footer
    """
    if not sources:
        return response

    # Add subtle footer with link icons
    footer = "\n\n" + " ".join([
        f"[🔗]({url})"  # Just icon, no text
        for _, url in sources
    ])

    return response + footer
```

**Example output**:

```
User: "When did I start the Lattice project?"

Bot: "You started working on Lattice in early January, around the 6th.
You mentioned it was inspired by the ENGRAM memory framework.

🔗 🔗 🔗"
```

Each 🔗 is a clickable Discord link (jump_url) to the source message.

### Integration Point

**Location**: `bot.py:301-308` (before `channel.send()`)

```python
# Generate response
response_result, rendered_prompt, context_info = await response_generator.generate_response(...)

# Split for Discord length limits
response_messages = response_generator.split_response(response_result.content)

# NEW: Inject source links
source_links = build_source_links(recent_messages, graph_triples)
response_messages[-1] = inject_source_links(response_messages[-1], source_links)

# Send to Discord
for msg in response_messages:
    bot_msg = await message.channel.send(msg)
```

### Building Source Map

```python
def build_source_links(
    recent_messages: list[EpisodicMessage],
    graph_triples: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Build list of (preview, jump_url) for sources used in context.

    Args:
        recent_messages: Messages used in episodic context
        graph_triples: Triples used in semantic context

    Returns:
        List of (preview, url) tuples for source attribution
    """
    sources = []

    # Add episodic sources (limit to last 5 for brevity)
    for msg in recent_messages[-5:]:
        if hasattr(msg, 'jump_url'):  # Bot messages have jump_url
            preview = msg.content[:30] + "..."
            sources.append((preview, msg.jump_url))

    # Add semantic sources (if triples have origin_id)
    # TODO: Fetch jump_url from raw_messages via origin_id FK
    for triple in graph_triples[:3]:  # Limit to 3 facts
        if 'origin_id' in triple:
            origin_url = await get_message_jump_url(triple['origin_id'])
            if origin_url:
                preview = f"{triple['subject']} {triple['predicate']} {triple['object']}"
                sources.append((preview, origin_url))

    return sources
```

### Fetching Jump URLs for Triples

**Challenge**: `semantic_triples.origin_id` links to `raw_messages(id)`, but we need Discord `jump_url`.

**Solution**: Store `jump_url` in `raw_messages` or reconstruct it.

```python
async def get_message_jump_url(message_id: UUID) -> str | None:
    """Get Discord jump URL for a raw_message.

    Args:
        message_id: UUID from raw_messages.id

    Returns:
        Discord jump URL or None
    """
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT discord_message_id, channel_id FROM raw_messages WHERE id = $1",
            message_id
        )

    if not row:
        return None

    # Reconstruct jump URL
    guild_id = os.getenv("DISCORD_GUILD_ID")  # Need to add to .env
    return f"https://discord.com/channels/{guild_id}/{row['channel_id']}/{row['discord_message_id']}"
```

**Required change**: Add `DISCORD_GUILD_ID` to `.env`.

## Summary & Recommendations

### Concurrency Model
✅ **Current design is good**
- Discord.py handles concurrency automatically
- Async fire-and-forget consolidation is correct choice
- No queue needed (yet)

### Consolidation Timing
✅ **Call AFTER send is optimal**
- Minimizes user-perceived latency
- Allows bot to stay responsive
- Acceptable tradeoff (eventual consistency)

### Triple Extraction
✅ **Ready to re-enable**
- Issue #61 complete (#87 merged)
- Schema ready (entities + semantic_triples exist)
- Just need to uncomment and implement `store_semantic_triples()`

### Link Injection
✅ **Implement minimal icon-based footer**
- Non-intrusive emoji links (🔗)
- Add at end of last message chunk
- Requires storing/reconstructing jump URLs

### Action Items

1. **Re-enable triple storage** (high priority)
   - [ ] Implement `store_semantic_triples()`
   - [ ] Implement `upsert_entity()`
   - [ ] Uncomment storage call in `episodic.py:237`
   - [ ] Add tests

2. **Implement link injection** (medium priority)
   - [ ] Add `DISCORD_GUILD_ID` to `.env`
   - [ ] Implement `get_message_jump_url()`
   - [ ] Implement `inject_source_links()`
   - [ ] Integrate in `bot.py:301-308`

3. **Address objective matching** (see issue #88)
   - [ ] Implement entity-based objective matching
   - [ ] Test with activity updates
   - [ ] Monitor false positives/negatives
