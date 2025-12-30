# How to View Lattice Bot Logs

## Option 1: Docker Compose (Recommended)

### Start the bot
```bash
make docker-up
```

### View all logs (bot + database)
```bash
make docker-logs
```

### View bot logs only
```bash
make docker-logs-bot
```

### View database logs only
```bash
make docker-logs-db
```

### Follow logs in real-time (Ctrl+C to stop)
```bash
docker compose logs -f bot
```

## Option 2: Local Development (Without Docker)

### Prerequisites
- PostgreSQL running locally on port 5432
- .env file configured with credentials

### Initialize database
```bash
export PATH="/home/chris/.local/bin:$PATH"
eval "$(mise activate bash)"
poetry run python scripts/init_db.py
```

### Run bot with logs to console
```bash
poetry run python -m lattice
```

Logs will appear in stdout/stderr

### Run bot with logs to file
The bot automatically writes to `logs/lattice.log` (configured in .env)

```bash
# Run bot in background
nohup poetry run python -m lattice > logs/bot-stdout.log 2>&1 &

# View logs
tail -f logs/lattice.log
tail -f logs/bot-stdout.log
```

## Log Configuration

Edit `.env` to configure logging:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Enable structured JSON logs (recommended for production)
STRUCTURED_LOGS=true

# Log file location
LOG_FILE=logs/lattice.log
```

## Common Log Entries to Look For

### Bot startup
```
Bot setup starting
Loading embedding model
Bot setup complete
Bot connected to Discord
```

### Message processing
```
Received message (author=username, content_preview=...)
Stored episodic message
Semantic search completed
Response sent successfully
```

### Errors
```
ERROR: Failed to connect to database
Error processing message (error=...)
```

## Debugging Discord Issues

If bot isn't responding to messages:

1. **Check Discord connection**:
   ```bash
   docker compose logs bot | grep "Bot connected to Discord"
   ```

2. **Verify channel ID**:
   ```bash
   echo $DISCORD_MAIN_CHANNEL_ID  # Should match your Discord channel
   ```

3. **Check for errors**:
   ```bash
   docker compose logs bot | grep ERROR
   ```

4. **Verify bot intents**:
   - Go to Discord Developer Portal > Your App > Bot
   - Enable "MESSAGE CONTENT INTENT" (privileged intent)
   - Save changes

5. **Check bot permissions in Discord**:
   - Bot needs "Read Messages", "Send Messages", "Read Message History"

## Testing the Bot

Send a message in the configured channel to test:

```
Hello bot!
```

Expected response:
```
I received your message: 'Hello bot!'. (Phase 1: Memory system operational, LLM integration coming in Phase 2)
```

## Stopping the Bot

### Docker:
```bash
make docker-down
```

### Local:
```bash
# Find process
ps aux | grep "python -m lattice"

# Kill process
kill <PID>
```
