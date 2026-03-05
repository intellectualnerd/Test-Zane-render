#!/usr/bin/env python3
"""
Example WebSocket client for QueryGuard Chat
This demonstrates how to connect to and interact with the WebSocket chat API.
"""

import asyncio
import json
import websockets
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryGuardChatClient:
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.websocket = None
        self.session_id = None
        self.org_id = None
        self.user_id = None
        self.user_name = None
        self.connected = False
        
    async def connect(self, org_id: str, user_id: str, user_name: str = None):
        """Connect to the WebSocket chat endpoint"""
        try:
            self.org_id = org_id
            self.user_id = user_id
            self.user_name = user_name or f"User_{user_id}"
            self.session_id = str(uuid.uuid4())
            
            # Construct WebSocket URL
            url = f"{self.base_url}/chat/ws/{org_id}/{user_id}"
            params = f"session_id={self.session_id}&user_name={self.user_name}"
            full_url = f"{url}?{params}"
            
            logger.info(f"Connecting to: {full_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(full_url)
            self.connected = True
            
            logger.info(f"Connected successfully! Session ID: {self.session_id}")
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def _message_listener(self):
        """Listen for incoming messages from the WebSocket"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Message listener error: {str(e)}")
            self.connected = False
    
    async def _handle_message(self, data):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type", "unknown")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())
        
        if message_type == "system_message":
            message = data.get("data", {}).get("message", "")
            print(f"[{timestamp}] SYSTEM: {message}")
            
        elif message_type == "chat_message":
            sender_name = data.get("data", {}).get("sender_name", "Unknown")
            content = data.get("data", {}).get("content", "")
            print(f"[{timestamp}] {sender_name}: {content}")
            
        elif message_type == "ai_response":
            response = data.get("data", {}).get("response", "")
            processing_time = data.get("data", {}).get("processing_time", 0)
            sources_count = len(data.get("data", {}).get("sources", []))
            print(f"[{timestamp}] AI ASSISTANT: {response}")
            print(f"    (Processing time: {processing_time:.2f}s, Sources: {sources_count})")
            
        elif message_type == "typing":
            sender_name = data.get("data", {}).get("sender_name", "Unknown")
            is_typing = data.get("data", {}).get("is_typing", False)
            if is_typing:
                print(f"{sender_name} is typing...")
            
        elif message_type == "user_status":
            message = data.get("data", {}).get("message", "")
            print(f"[{timestamp}] STATUS: {message}")
            
        elif message_type == "error":
            error_message = data.get("data", {}).get("error_message", "Unknown error")
            print(f"[{timestamp}] ERROR: {error_message}")
            
        elif message_type == "pong":
            print("Received pong from server")
            
        else:
            print(f"[{timestamp}] UNKNOWN MESSAGE TYPE ({message_type}): {data}")
    
    async def send_message(self, content: str, conversation_id: str = None):
        """Send a chat message"""
        if not self.connected or not self.websocket:
            logger.error("Not connected to WebSocket")
            return False
            
        try:
            message = {
                "type": "chat_message",
                "content": content,
                "conversation_id": conversation_id
            }
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return False
    
    async def send_typing(self, is_typing: bool = True):
        """Send typing indicator"""
        if not self.connected or not self.websocket:
            return False
            
        try:
            message = {
                "type": "typing",
                "data": {"is_typing": is_typing}
            }
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending typing indicator: {str(e)}")
            return False
    
    async def ping(self):
        """Send a ping to keep connection alive"""
        if not self.connected or not self.websocket:
            return False
            
        try:
            message = {"type": "ping"}
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending ping: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.connected = False
                logger.info("Disconnected from WebSocket")
                
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")

async def interactive_chat_session():
    """Run an interactive chat session"""
    client = QueryGuardChatClient()
    
    # Get connection details from user
    print("QueryGuard WebSocket Chat Client")
    print("=" * 40)
    
    org_id = input("Enter Organization ID (default: 76d33fb3-6062-456b-a211-4aec9971f8be): ").strip()
    if not org_id:
        org_id = "76d33fb3-6062-456b-a211-4aec9971f8be"
    
    user_id = input("Enter User ID (default: test-user): ").strip()
    if not user_id:
        user_id = "test-user"
    
    user_name = input("Enter User Name (default: Test User): ").strip()
    if not user_name:
        user_name = "Test User"
    
    print(f"\\nConnecting to chat as {user_name} (ID: {user_id}) in org {org_id}...")
    
    # Connect to WebSocket
    connected = await client.connect(org_id, user_id, user_name)
    if not connected:
        print("Failed to connect to WebSocket")
        return
    
    print("Connected! Type messages and press Enter to send.")
    print("Special commands:")
    print("  /ping - Send ping to server")
    print("  /typing - Send typing indicator")
    print("  /quit - Exit chat")
    print("=" * 40)
    
    # Chat loop
    try:
        while client.connected:
            try:
                # Get user input
                message = await asyncio.to_thread(input, "> ")
                message = message.strip()
                
                if not message:
                    continue
                
                # Handle special commands
                if message == "/quit":
                    break
                elif message == "/ping":
                    await client.ping()
                    continue
                elif message == "/typing":
                    await client.send_typing(True)
                    await asyncio.sleep(1)
                    await client.send_typing(False)
                    continue
                
                # Send regular message
                await client.send_message(message)
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                
    finally:
        await client.disconnect()
        print("Chat session ended.")

async def automated_test():
    """Run automated test with predefined messages"""
    client = QueryGuardChatClient()
    
    org_id = "76d33fb3-6062-456b-a211-4aec9971f8be"
    user_id = "test-user-auto"
    user_name = "Automated Test User"
    
    print(f"Running automated test for {user_name}...")
    
    # Connect
    connected = await client.connect(org_id, user_id, user_name)
    if not connected:
        print("Failed to connect")
        return
    
    # Wait for connection to settle
    await asyncio.sleep(2)
    
    # Send test messages
    test_messages = [
        "Hello, this is an automated test!",
        "What is data lineage?",
        "Can you explain impact analysis?",
        "Show me information about Snowflake connections"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\\nSending test message {i}/{len(test_messages)}: {message}")
        
        # Send typing indicator
        await client.send_typing(True)
        await asyncio.sleep(1)
        await client.send_typing(False)
        
        # Send message
        await client.send_message(message)
        
        # Wait for response
        await asyncio.sleep(5)
    
    # Send ping test
    print("\\nTesting ping...")
    await client.ping()
    
    # Wait a bit more
    await asyncio.sleep(3)
    
    # Disconnect
    await client.disconnect()
    print("Automated test completed.")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run automated test
        asyncio.run(automated_test())
    else:
        # Run interactive session
        asyncio.run(interactive_chat_session())

if __name__ == "__main__":
    main()