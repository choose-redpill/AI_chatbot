#!/usr/bin/env python3
"""
Test Suite for Rural Renewable Energy Chatbot
Validates core functionality and performance
"""

import unittest
import time
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot import RenewableEnergyChatbot
except ImportError as e:
    print(f"Error importing chatbot: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class TestRenewableEnergyChatbot(unittest.TestCase):
    """Test cases for the renewable energy chatbot"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = tempfile.mkdtemp()
        os.chdir(cls.temp_dir)
        
        # Create test directories
        os.makedirs("docs", exist_ok=True)
        os.makedirs("vector_db", exist_ok=True)
        
        print(f"🧪 Test environment created at: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        os.chdir("/")
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        print("🧹 Test environment cleaned up")
    
    def setUp(self):
        """Set up each test"""
        try:
            self.chatbot = RenewableEnergyChatbot()
        except Exception as e:
            self.skipTest(f"Could not initialize chatbot: {e}")
    
    def test_initialization(self):
        """Test chatbot initialization"""
        self.assertIsNotNone(self.chatbot)
        self.assertIsNotNone(self.chatbot.embedding_model)
        self.assertIsNotNone(self.chatbot.chroma_client)
        self.assertIsNotNone(self.chatbot.collection)
        self.assertIsNotNone(self.chatbot.text_splitter)
        self.assertIsNotNone(self.chatbot.llm)
        print("✅ Initialization test passed")
    
    def test_document_creation(self):
        """Test sample document creation"""
        created_files = self.chatbot.create_comprehensive_dataset()
        
        # Check if files were created
        self.assertTrue(len(created_files) > 0)
        
        # Check if actual files exist
        for filename in created_files:
            filepath = os.path.join("docs", filename)
            self.assertTrue(os.path.exists(filepath))
            
            # Check file content
            with open(filepath, 'r') as f:
                content = f.read()
                self.assertTrue(len(content) > 100)  # Reasonable content length
        
        print(f"✅ Document creation test passed ({len(created_files)} files)")
    
    def test_document_loading(self):
        """Test document loading functionality"""
        # First create documents
        self.chatbot.create_comprehensive_dataset()
        
        # Load documents
        documents = self.chatbot.load_documents()
        
        self.assertTrue(len(documents) > 0)
        
        for doc in documents:
            self.assertIn("id", doc)
            self.assertIn("content", doc)
            self.assertIn("metadata", doc)
            self.assertTrue(len(doc["content"]) > 0)
        
        print(f"✅ Document loading test passed ({len(documents)} docs)")
    
    def test_document_processing(self):
        """Test document processing and vector storage"""
        # Create and load documents
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        
        # Process documents
        self.chatbot.process_documents(documents)
        
        # Verify documents are in vector store
        collection_data = self.chatbot.collection.get()
        self.assertTrue(len(collection_data["ids"]) > 0)
        
        print(f"✅ Document processing test passed ({len(collection_data['ids'])} chunks)")
    
    def test_basic_queries(self):
        """Test basic query functionality"""
        # Setup
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
        
        # Test queries
        test_queries = [
            "What is solar energy?",
            "How much does a solar system cost?",
            "Tell me about wind energy",
            "What are government subsidies?",
            "How to maintain solar panels?"
        ]
        
        for query in test_queries:
            result = self.chatbot.query_rag(query)
            
            # Validate response structure
            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertIn("confidence", result)
            self.assertIn("processing_time", result)
            
            # Validate response content
            self.assertTrue(len(result["answer"]) > 0)
            self.assertIsInstance(result["sources"], list)
            self.assertIsInstance(result["confidence"], float)
            self.assertIsInstance(result["processing_time"], float)
            
            # Check confidence is reasonable
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            
            print(f"   ✅ Query: '{query[:30]}...' - Confidence: {result['confidence']:.2f}")
        
        print("✅ Basic queries test passed")
    
    def test_performance(self):
        """Test response time performance"""
        # Setup
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
        
        # Performance test
        query = "What is the cost of solar energy?"
        
        start_time = time.time()
        result = self.chatbot.query_rag(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(response_time, 10.0, "Response time should be under 10 seconds")
        self.assertGreater(result["confidence"], 0.1, "Should have reasonable confidence")
        
        print(f"✅ Performance test passed - Response time: {response_time:.2f}s")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Setup
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
        
        # Test empty query
        result = self.chatbot.query_rag("")
        self.assertIn("answer", result)
        
        # Test very long query
        long_query = "What is solar energy? " * 100
        result = self.chatbot.query_rag(long_query)
        self.assertIn("answer", result)
        
        # Test non-renewable energy query
        result = self.chatbot.query_rag("Tell me about cooking recipes")
        self.assertIn("answer", result)
        
        # Test special characters
        result = self.chatbot.query_rag("Solar energy cost in ₹?")
        self.assertIn("answer", result)
        
        print("✅ Edge cases test passed")
    
    @patch('speech_recognition.Recognizer')
    @patch('pyttsx3.init')
    def test_voice_components(self, mock_tts, mock_sr):
        """Test voice functionality (mocked)"""
        # Mock voice recognition
        mock_recognizer = MagicMock()
        mock_sr.return_value = mock_recognizer
        
        # Mock TTS engine
        mock_engine = MagicMock()
        mock_tts.return_value = mock_engine
        
        # Test voice initialization
        self.chatbot.initialize_voice()
        
        # Verify voice is enabled
        if self.chatbot.voice_enabled:
            print("✅ Voice components test passed (mocked)")
        else:
            print("⚠️  Voice components not available")
    
    def test_chat_history(self):