#!/usr/bin/env python3
"""
Test script to verify WebSocket components work correctly
"""

import sys
import os
sys.path.append('.')

def test_websocket_components():
    """Test WebSocket components without database dependencies"""
    print("🧪 Testing WebSocket Components...")
    
    try:
        # Test basic imports
        from app.utils.websocket_manager import WebSocketManager
        print("✅ WebSocket manager imported successfully")
        
        # Test Pydantic models
        from app.utils.websocket_models import (
            WebSocketMessageData,
            MessageType,
            ChatSessionInfo,
            UserStatus,
            ChatStatsResponse
        )
        print("✅ WebSocket models imported successfully")
        
        # Test creating WebSocket message
        msg = WebSocketMessageData(
            type=MessageType.CHAT_MESSAGE,
            data={"content": "Hello, World!"},
            sender_id="test-user"
        )
        print(f"✅ WebSocket message created: {msg.type} from {msg.sender_id}")
        
        # Test WebSocket manager
        manager = WebSocketManager()
        stats = manager.get_stats()
        print(f"✅ WebSocket manager created, initial stats: {stats}")
        
        # Test message types enum
        print("✅ Available message types:")
        for msg_type in MessageType:
            print(f"   - {msg_type.value}")
            
        # Test session info model
        session_info = ChatSessionInfo(
            session_id="test-session",
            org_id="test-org",
            user_id="test-user",
            status=UserStatus.ONLINE,
            created_at="2023-11-07T10:00:00Z",
            last_activity="2023-11-07T10:00:00Z"
        )
        print(f"✅ Chat session info created: {session_info.session_id}")
        
        # Test stats response
        stats_response = ChatStatsResponse(
            active_connections=0,
            active_sessions=0,
            organizations_with_sessions=0,
            users_with_sessions=0,
            total_org_sessions=0,
            total_user_sessions=0
        )
        print(f"✅ Chat stats response created: {stats_response.active_connections} connections")
        
        print("\\n🎉 All WebSocket components are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing WebSocket components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_message_serialization():
    """Test message serialization/deserialization"""
    print("\\n🧪 Testing Message Serialization...")
    
    try:
        from app.utils.websocket_models import WebSocketMessageData, MessageType
        import json
        
        # Create test message
        msg = WebSocketMessageData(
            type=MessageType.CHAT_MESSAGE,
            data={"content": "Test message", "conversation_id": "test-conv"},
            sender_id="user123"
        )
        
        # Serialize to dict
        msg_dict = msg.model_dump()
        print(f"✅ Message serialized to dict: {msg_dict['type']}")
        
        # Serialize to JSON
        msg_json = json.dumps(msg_dict)
        print(f"✅ Message serialized to JSON: {len(msg_json)} characters")
        
        # Deserialize from JSON
        msg_dict_restored = json.loads(msg_json)
        msg_restored = WebSocketMessageData(**msg_dict_restored)
        print(f"✅ Message deserialized from JSON: {msg_restored.type}")
        
        # Verify equality
        assert msg.type == msg_restored.type
        assert msg.sender_id == msg_restored.sender_id
        assert msg.data == msg_restored.data
        print("✅ Serialization round-trip successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing message serialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting WebSocket Component Tests\\n")
    
    # Run tests
    test1_result = test_websocket_components()
    test2_result = test_message_serialization()
    
    # Summary
    print("\\n" + "="*50)
    if test1_result and test2_result:
        print("🎉 ALL TESTS PASSED! WebSocket implementation is ready!")
        print("\\n📋 Next Steps:")
        print("1. Set up your database environment variables")
        print("2. Start the FastAPI server: uvicorn app.main:app --reload")
        print("3. Test WebSocket at: http://localhost:8000/chat/test-page")
        print("4. Or use the Python client: python websocket_client_example.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)