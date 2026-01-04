# Dream Channel Integration Testing Guide

## Prerequisites

1. **Discord Setup:**
   - Have a Discord server with two text channels
   - Note the channel IDs:
     - Main channel (for conversation)
     - Dream channel (for prompt audits and feedback)

2. **Environment Configuration:**
   ```bash
   # Add to .env file
   DISCORD_MAIN_CHANNEL_ID=<your_main_channel_id>
   DISCORD_DREAM_CHANNEL_ID=<your_dream_channel_id>
   ```

3. **Start Services:**
   ```bash
   make docker-up
   make migrate
   ```

## Test Scenarios

### Test 1: Basic Message Flow (Prompt Audit + Dream Mirror)

**Actions:**
1. Send a message in main channel: "Hello, what's the weather like?"
2. Wait for bot response

**Expected Results:**
- ✅ Bot responds in main channel
- ✅ Dream channel receives mirrored message with:
  - 🔗 Jump link to main message
  - Full bot response
  - Metadata line (prompt key, context counts, latency, cost)
  - "💬 Reply or quote this message for feedback"
  - Spoiler-wrapped full prompt

**Verification:**
```sql
-- Check prompt_audits table
SELECT id, prompt_key, main_discord_message_id, dream_discord_message_id
FROM prompt_audits
ORDER BY created_at DESC
LIMIT 1;
```

### Test 2: Feedback in Dream Channel (Link to Audit)

**Actions:**
1. In dream channel, reply to the mirrored message
2. Type: "Great explanation, very clear!"

**Expected Results:**
- ✅ Bot reacts with 🫡 emoji on your feedback message
- ✅ Feedback is stored in `user_feedback` table
- ✅ Feedback is linked to prompt audit

**Verification:**
```sql
-- Check feedback is linked to audit
SELECT
    pa.id AS audit_id,
    pa.prompt_key,
    pa.dream_discord_message_id,
    uf.id AS feedback_id,
    uf.content AS feedback_content
FROM prompt_audits pa
JOIN user_feedback uf ON pa.feedback_id = uf.id
ORDER BY pa.created_at DESC
LIMIT 1;
```

### Test 3: Feedback in Main Channel (Should NOT Work)

**Actions:**
1. In main channel, reply to bot's original message
2. Type: "This is wrong!"

**Expected Results:**
- ❌ NO 🫡 emoji reaction
- ❌ Feedback is NOT stored
- ✅ Bot treats it as a regular message

**Verification:**
```sql
-- Should NOT find feedback for main channel message
SELECT COUNT(*)
FROM user_feedback
WHERE user_discord_message_id = <your_feedback_message_id>;
-- Result should be 0
```

### Test 4: Multiple Responses (Split Message Handling)

**Actions:**
1. Send a message that triggers a long response (>2000 chars)
2. Example: "Tell me everything about quantum physics in detail"

**Expected Results:**
- ✅ Bot sends multiple messages in main channel (if response is long)
- ✅ Each response gets its own prompt audit
- ✅ Each response gets mirrored to dream channel
- ✅ All audits reference same input message

**Verification:**
```sql
-- Check multiple audits for same input
SELECT COUNT(*), message_id
FROM prompt_audits
WHERE message_id = (
    SELECT id FROM raw_messages
    WHERE is_bot = false
    ORDER BY timestamp DESC
    LIMIT 1
)
GROUP BY message_id;
```

### Test 5: Feedback Undo (Delete Feedback)

**Actions:**
1. Give feedback in dream channel (as in Test 2)
2. Wait for 🫡 reaction
3. React to your feedback message with 🗑️ emoji

**Expected Results:**
- ✅ 🫡 emoji is removed
- ✅ Feedback is deleted from database
- ✅ Audit's feedback_id is set back to NULL

**Verification:**
```sql
-- Check feedback was unlinked
SELECT feedback_id
FROM prompt_audits
WHERE dream_discord_message_id = <dream_message_id>;
-- Should be NULL
```

### Test 6: Context Configuration Tracking

**Actions:**
1. Send various types of messages
2. Check that context config is stored correctly

**Verification:**
```sql
-- Check context_config is populated
SELECT
    prompt_key,
    context_config,
    created_at
FROM prompt_audits
ORDER BY created_at DESC
LIMIT 5;
```

Should show JSON like:
```json
{
  "episodic": 5,
  "semantic": 3,
  "graph": 0
}
```

### Test 7: Long Prompt Truncation (Spoiler Content)

**Actions:**
1. Have a conversation that builds up context
2. Send a message that will use lots of context
3. Check dream channel message

**Expected Results:**
- ✅ If prompt > 1500 chars, it's truncated with "... (truncated)"
- ✅ Spoiler tags work correctly in Discord UI
- ✅ Full untruncated prompt is in database

**Verification:**
```sql
-- Compare rendered prompt length in DB vs displayed
SELECT
    prompt_key,
    LENGTH(rendered_prompt) AS full_length,
    rendered_prompt
FROM prompt_audits
ORDER BY created_at DESC
LIMIT 1;
```

## Database Queries for Analysis

### Get All Audits with Feedback
```sql
SELECT
    pa.id,
    pa.prompt_key,
    pa.created_at,
    uf.content AS feedback,
    pa.context_config
FROM prompt_audits pa
JOIN user_feedback uf ON pa.feedback_id = uf.id
ORDER BY pa.created_at DESC;
```

### Feedback Rate Analysis
```sql
SELECT
    COUNT(*) AS total_audits,
    COUNT(feedback_id) AS audits_with_feedback,
    ROUND(COUNT(feedback_id)::numeric / COUNT(*)::numeric * 100, 2) AS feedback_rate_percent
FROM prompt_audits;
```

### Average Performance Metrics
```sql
SELECT
    prompt_key,
    AVG(latency_ms) AS avg_latency_ms,
    AVG(cost_usd) AS avg_cost_usd,
    AVG(prompt_tokens) AS avg_prompt_tokens,
    COUNT(*) AS usage_count
FROM prompt_audits
GROUP BY prompt_key
ORDER BY usage_count DESC;
```

### Recent Interactions Timeline
```sql
SELECT
    rm.timestamp,
    rm.is_bot,
    LEFT(rm.content, 50) AS content_preview,
    pa.prompt_key,
    pa.latency_ms,
    CASE
        WHEN pa.feedback_id IS NOT NULL THEN '✅ Has feedback'
        ELSE ''
    END AS feedback_status
FROM raw_messages rm
LEFT JOIN prompt_audits pa ON rm.id = pa.message_id
ORDER BY rm.timestamp DESC
LIMIT 20;
```

## Troubleshooting

### Issue: No Dream Channel Mirror

**Possible Causes:**
1. DISCORD_DREAM_CHANNEL_ID not set in .env
2. Bot doesn't have permissions in dream channel
3. Dream channel doesn't exist

**Solution:**
```bash
# Check logs
docker logs lattice-bot --tail 100 | grep -i dream

# Verify environment variable
docker exec lattice-bot env | grep DREAM
```

### Issue: Feedback Not Linked to Audit

**Possible Causes:**
1. Feedback given in wrong channel (main instead of dream)
2. Dream message ID not matching

**Solution:**
```sql
-- Find orphaned feedback (not linked to any audit)
SELECT
    uf.id,
    uf.content,
    uf.referenced_discord_message_id,
    uf.created_at
FROM user_feedback uf
LEFT JOIN prompt_audits pa ON pa.feedback_id = uf.id
WHERE pa.id IS NULL;
```

### Issue: Database Pool Errors

**Solution:**
```bash
# Restart services
make docker-restart

# Check database connectivity
docker exec lattice-postgres psql -U lattice -d lattice -c "SELECT COUNT(*) FROM prompt_audits;"
```

## Success Criteria

All tests pass when:
- [x] Prompt audits stored for every bot response
- [x] Dream channel receives mirrors with correct format
- [x] Feedback only works in dream channel
- [x] Feedback correctly linked to audits
- [x] Main channel remains clean (no audit noise)
- [x] Database has complete audit trail
- [x] Tests can be reproduced consistently
