#!/usr/bin/env python3
"""
Test script to debug VK bot initialization
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

async def test_vk_bot():
    try:
        from app.vk_bot import create_vk_bot
        from app.database import TicketsSessionLocal
        from app.realtime import ConnectionManager

        logger.info("Testing VK bot initialization...")

        # Create connection manager
        connection_manager = ConnectionManager()
        logger.info("Connection manager created")

        # Create a mock RAG service
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

        # Get VK token
        vk_token = os.getenv("VK_ACCESS_TOKEN")
        logger.info(f"VK token found: {'YES' if vk_token else 'NO'}")

        if vk_token:
            logger.info("Creating VK bot...")
            try:
                vk_bot = create_vk_bot(TicketsSessionLocal, connection_manager, rag_service, vk_token)
                if vk_bot:
                    logger.info("VK bot created successfully!")
                else:
                    logger.error("VK bot creation failed - returned None")
            except Exception as e:
                logger.exception(f"VK bot creation failed with exception: {e}")
        else:
            logger.error("No VK token found")

    except Exception as e:
        logger.exception(f"Error during VK bot test: {e}")

if __name__ == "__main__":
    asyncio.run(test_vk_bot())