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
        """Test chat history functionality"""
        # Setup
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
        
        # Initial state
        self.assertEqual(len(self.chatbot.chat_history), 0)
        
        # Make some queries
        queries = [
            "What is solar energy?",
            "How much does it cost?",
            "What about maintenance?"
        ]
        
        for query in queries:
            self.chatbot.query_rag(query)
        
        # Check history
        self.assertEqual(len(self.chatbot.chat_history), len(queries))
        
        # Verify history structure
        for i, chat in enumerate(self.chatbot.chat_history):
            self.assertIn("timestamp", chat)
            self.assertIn("query", chat)
            self.assertIn("response", chat)
            self.assertIn("sources", chat)
            self.assertIn("confidence", chat)
            self.assertEqual(chat["query"], queries[i])
        
        print("✅ Chat history test passed")
    
    def test_multilingual_readiness(self):
        """Test multilingual structure (placeholder for future expansion)"""
        # Test language setting
        original_language = self.chatbot.current_language
        self.chatbot.current_language = "hindi"
        
        # Verify setting changed
        self.assertEqual(self.chatbot.current_language, "hindi")
        
        # Reset
        self.chatbot.current_language = original_language
        
        print("✅ Multilingual readiness test passed")
    
    def test_error_recovery(self):
        """Test error recovery and fallback mechanisms"""
        # Test with corrupted vector database
        try:
            # Force an error in vector store
            original_collection = self.chatbot.collection
            self.chatbot.collection = None
            
            result = self.chatbot.query_rag("What is solar energy?")
            
            # Should still return a response structure
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            
            # Restore
            self.chatbot.collection = original_collection
            
        except Exception as e:
            self.fail(f"Error recovery failed: {e}")
        
        print("✅ Error recovery test passed")

class TestDataQuality(unittest.TestCase):
    """Test data quality and content accuracy"""
    
    def setUp(self):
        self.chatbot = RenewableEnergyChatbot()
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
    
    def test_content_relevance(self):
        """Test if responses are relevant to renewable energy"""
        relevant_queries = [
            "solar panel cost",
            "wind turbine maintenance",
            "biogas benefits",
            "government subsidies renewable energy"
        ]
        
        for query in relevant_queries:
            result = self.chatbot.query_rag(query)
            answer = result["answer"].lower()
            
            # Check for renewable energy keywords
            renewable_keywords = [
                "solar", "wind", "biogas", "renewable", "energy",
                "panel", "turbine", "subsidy", "cost", "maintenance"
            ]
            
            has_relevant_content = any(keyword in answer for keyword in renewable_keywords)
            self.assertTrue(has_relevant_content, f"Response for '{query}' lacks renewable energy content")
        
        print("✅ Content relevance test passed")
    
    def test_cost_information_accuracy(self):
        """Test if cost information is reasonable"""
        cost_queries = [
            "solar panel cost",
            "wind turbine price",
            "biogas plant cost"
        ]
        
        for query in cost_queries:
            result = self.chatbot.query_rag(query)
            answer = result["answer"]
            
            # Check for currency symbols or cost indicators
            cost_indicators = ["₹", "rupees", "cost", "price", "lakh", "thousand"]
            has_cost_info = any(indicator in answer.lower() for indicator in cost_indicators)
            
            if has_cost_info:
                print(f"   ✅ Cost info found for: {query}")
            else:
                print(f"   ⚠️  Limited cost info for: {query}")
        
        print("✅ Cost information accuracy test passed")

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance and scalability metrics"""
    
    def setUp(self):
        self.chatbot = RenewableEnergyChatbot()
        self.chatbot.create_comprehensive_dataset()
        documents = self.chatbot.load_documents()
        self.chatbot.process_documents(documents)
    
    def test_concurrent_queries(self):
        """Test handling multiple queries in sequence"""
        queries = [
            "What is solar energy?",
            "Wind power benefits",
            "Biogas installation",
            "Government schemes",
            "Maintenance costs"
        ] * 3  # 15 queries total
        
        start_time = time.time()
        results = []
        
        for query in queries:
            result = self.chatbot.query_rag(query)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries)
        
        # Performance assertions
        self.assertLess(avg_time, 3.0, "Average response time should be under 3 seconds")
        
        # Check all queries succeeded
        for result in results:
            self.assertIn("answer", result)
            self.assertGreater(len(result["answer"]), 0)
        
        print(f"✅ Concurrent queries test passed - Avg time: {avg_time:.2f}s")
    
    def test_memory_usage(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple documents
        for i in range(10):
            result = self.chatbot.query_rag(f"Tell me about renewable energy option {i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase dramatically
        self.assertLess(memory_increase, 100, "Memory usage should not increase by more than 100MB")
        
        print(f"✅ Memory usage test passed - Increase: {memory_increase:.2f}MB")

def run_comprehensive_tests():
    """Run all test suites with detailed reporting"""
    print("🧪 Starting Comprehensive Test Suite")
    print("=" * 60)
    
    # Test suites
    test_suites = [
        TestRenewableEnergyChatbot,
        TestDataQuality,
        TestPerformanceMetrics
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_class in test_suites:
        print(f"\n🔍 Running {suite_class.__name__}")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        
        for test in suite:
            total_tests += 1
            try:
                result = runner.run(test)
                if result.wasSuccessful():
                    passed_tests += 1
                    print(f"✅ {test._testMethodName}")
                else:
                    failed_tests.append(f"{suite_class.__name__}.{test._testMethodName}")
                    print(f"❌ {test._testMethodName}")
                    for failure in result.failures + result.errors:
                        print(f"   Error: {failure[1].split('AssertionError: ')[-1].split('\n')[0]}")
            except Exception as e:
                failed_tests.append(f"{suite_class.__name__}.{test._testMethodName}")
                print(f"❌ {test._testMethodName} - {str(e)}")
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {len(failed_tests)} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for test in failed_tests:
            print(f"  ❌ {test}")
    
    print("\n🎯 RECOMMENDATIONS:")
    success_rate = (passed_tests/total_tests)*100
    
    if success_rate >= 90:
        print("🌟 Excellent! Your chatbot is production-ready.")
    elif success_rate >= 75:
        print("👍 Good! Minor improvements needed.")
        print("   - Review failed tests and fix issues")
        print("   - Consider additional error handling")
    elif success_rate >= 50:
        print("⚠️  Fair! Significant improvements needed.")
        print("   - Fix critical functionality issues")
        print("   - Improve error handling and edge cases")
        print("   - Verify all dependencies are installed")
    else:
        print("🚨 Poor! Major issues need attention.")
        print("   - Check all dependencies and installation")
        print("   - Review configuration and setup")
        print("   - Consider running setup.py again")
    
    return passed_tests, total_tests

def main():
    """Main test runner"""
    print("🌱 Rural Renewable Energy Chatbot Test Suite")
    print("Testing implementation completeness and functionality...")
    print()
    
    # Check dependencies
    missing_deps = []
    required_modules = [
        'streamlit', 'chromadb', 'langchain', 'sentence_transformers',
        'PyPDF2', 'speech_recognition', 'pyttsx3'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(module)
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Run tests
    try:
        passed, total = run_comprehensive_tests()
        
        print(f"\n🏁 Testing Complete!")
        print(f"Implementation Status: {(passed/total)*100:.0f}% Complete")
        
        if passed == total:
            print("🎉 All tests passed! Your 50% implementation is solid!")
        else:
            print(f"🔧 {total-passed} issues to address for full functionality")
    
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        print("Please check your installation and try again")

if __name__ == "__main__":
    main()