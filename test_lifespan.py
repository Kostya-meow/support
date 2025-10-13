#!/usr/bin/env python3
"""
Test lifespan script
"""
import os
import sys
import asyncio
import logging

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_lifespan():
    try:
        from app.database import init_db
        from app.realtime import ConnectionManager
        from app.config import load_config
        from app.vk_bot import create_vk_bot
        from app.database import TicketsSessionLocal

        logger.info("🚀 Starting test lifespan...")

        await init_db()
        connection_manager = ConnectionManager()

        # Create mock RAG service
        class MockRAGService:
            def get_chat_history_since_last_ticket(self, chat_id):
                return []
            def mark_ticket_created(self, chat_id):
                pass
            def reset_history(self, ticket_id):
                pass
            def add_to_history(self, conversation_id, message, is_user):
                pass
            def generate_ticket_summary(self, messages, ticket_id=None):
                return "Mock summary"
            def generate_reply(self, conversation_id, user_text):
                class MockResult:
                    operator_requested = False
                    final_answer = "Mock answer"
                return MockResult()

        rag_service = MockRAGService()
        logger.info("Mock RAG service created")

        # VK Bot
        vk_token = os.getenv("VK_ACCESS_TOKEN")
        logger.info(f"VK: Token found: {'YES' if vk_token else 'NO'}")
        if vk_token:
            logger.info("VK: Attempting to create VK bot...")
            vk_run_bot = create_vk_bot(TicketsSessionLocal, connection_manager, rag_service, vk_token)
            if vk_run_bot is not None:
                logger.info("VK: Bot created successfully, starting task...")
                # Start the bot task but don't wait for it
                vk_task = asyncio.create_task(vk_run_bot())
                logger.info("VK: Bot task started")
                # Let it run for a few seconds
                await asyncio.sleep(5)
                logger.info("VK: Cancelling bot task...")
                vk_task.cancel()
                try:
                    await vk_task
                except asyncio.CancelledError:
                    logger.info("VK: Bot task cancelled successfully")
            else:
                logger.warning("VK: Bot disabled due to configuration issues")
        else:
            logger.warning("VK_ACCESS_TOKEN is not set. VK integration is disabled.")

        logger.info("✅ Test lifespan completed")

    except Exception as e:
        logger.exception(f"Error during test lifespan: {e}")

if __name__ == "__main__":
    asyncio.run(test_lifespan())