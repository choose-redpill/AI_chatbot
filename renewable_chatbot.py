# Enhanced AI-powered Renewable Energy Chatbot for Rural Communities
# 50% Implementation with improved features and reliability

import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import speech_recognition as sr
import pyttsx3
import threading

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RenewableEnergyChatbot:
    """Enhanced chatbot class with better error handling and features"""
    
    def __init__(self):
        self.initialize_components()
        self.chat_history = []
        self.voice_enabled = False
        self.current_language = "english"
        
    def initialize_components(self):
        """Initialize all chatbot components with error handling"""
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embedding model loaded successfully")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="./vector_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="renewable_energy_enhanced"
            )
            logger.info("ChromaDB initialized successfully")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Initialize LLM with fallback
            self.initialize_llm()
            
            # Initialize voice components
            self.initialize_voice()
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def initialize_llm(self):
        """Initialize LLM with fallback options"""
        try:
            # Try to connect to Ollama first
            self.llm = Ollama(model="mistral", temperature=0.3)
            # Test the connection
            test_response = self.llm("Hello")
            logger.info("Ollama Mistral model connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {str(e)}")
            # Fallback to a mock LLM for demonstration
            self.llm = self.create_fallback_llm()
    
    def create_fallback_llm(self):
        """Create a fallback LLM for when Ollama is not available"""
        class FallbackLLM:
            def __call__(self, prompt):
                # Simple rule-based responses for demo purposes
                responses = {
                    "solar": "Solar energy harnesses sunlight using photovoltaic panels. It's ideal for rural areas due to low maintenance requirements and scalability. Government subsidies are available for installation.",
                    "wind": "Wind energy uses turbines to convert wind into electricity. Small-scale turbines are suitable for rural communities with consistent wind speeds above 3 m/s.",
                    "biogas": "Biogas is produced from organic waste through anaerobic digestion. It's perfect for rural areas with livestock as it provides both cooking gas and fertilizer.",
                    "maintenance": "Regular maintenance includes cleaning panels monthly, checking connections quarterly, and professional inspection annually.",
                    "cost": "Initial costs vary: Solar panels: ₹40,000-80,000 for 1kW system. Government subsidies can reduce costs by 30-70%.",
                    "installation": "Installation process: 1) Site assessment, 2) System design, 3) Permits and approvals, 4) Installation, 5) Grid connection and testing."
                }
                
                prompt_lower = prompt.lower()
                for key, response in responses.items():
                    if key in prompt_lower:
                        return f"Based on your query about renewable energy: {response}"
                
                return "I can help you with information about solar energy, wind power, biogas, maintenance, costs, and installation. Please ask a specific question about renewable energy for rural areas."
        
        logger.info("Using fallback LLM for demonstration")
        return FallbackLLM()
    
    def initialize_voice(self):
        """Initialize voice recognition and synthesis components"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            
            self.voice_enabled = True
            logger.info("Voice components initialized successfully")
        except Exception as e:
            logger.warning(f"Voice initialization failed: {str(e)}")
            self.voice_enabled = False
    
    def create_comprehensive_dataset(self):
        """Create a comprehensive renewable energy dataset"""
        renewable_energy_data = {
            "solar_energy_basics.txt": """
            Solar Energy for Rural Communities
            
            Solar energy is one of the most suitable renewable energy sources for rural areas in India. 
            Solar panels convert sunlight directly into electricity using photovoltaic cells.
            
            Benefits for Rural Areas:
            - Low maintenance requirements
            - Scalable from small home systems to community installations
            - Reduces dependence on grid electricity
            - Can power water pumps, lighting, and small appliances
            - Suitable for remote locations
            
            Types of Solar Systems:
            1. Grid-tied systems: Connected to the main electricity grid
            2. Off-grid systems: Independent systems with battery storage
            3. Hybrid systems: Combination of grid-tied and off-grid
            
            Cost Information:
            - 1kW solar system: ₹40,000 to ₹80,000
            - Government subsidies available up to 70% for residential systems
            - Payback period: 4-6 years
            - System lifespan: 25+ years
            """,
            
            "wind_energy_rural.txt": """
            Wind Energy Solutions for Rural Communities
            
            Small wind turbines can be an excellent renewable energy solution for rural areas 
            with consistent wind speeds above 3 meters per second.
            
            Suitable Locations:
            - Open areas with minimal obstructions
            - Coastal regions
            - Hilly terrains
            - Areas with consistent wind patterns
            
            Small Wind Turbine Options:
            - Micro turbines (under 1kW): ₹50,000 to ₹1,00,000
            - Small turbines (1-10kW): ₹1,00,000 to ₹8,00,000
            - Community turbines (10-50kW): ₹8,00,000 to ₹30,00,000
            
            Maintenance Requirements:
            - Monthly visual inspection
            - Quarterly lubrication of moving parts
            - Annual professional maintenance
            - Battery replacement every 3-5 years
            """,
            
            "biogas_systems.txt": """
            Biogas Technology for Rural Households
            
            Biogas is an ideal renewable energy solution for rural families with livestock.
            It converts organic waste into cooking gas and high-quality fertilizer.
            
            Benefits:
            - Free cooking gas from kitchen and animal waste
            - High-quality organic fertilizer (slurry)
            - Reduces indoor air pollution
            - Eliminates need for firewood collection
            - Reduces greenhouse gas emissions
            
            Types of Biogas Plants:
            1. Fixed Dome Plants: Most common, durable, underground construction
            2. Floating Gas Holder Plants: Above ground, easier maintenance
            3. Bag Type Plants: Low cost, portable, suitable for small families
            
            Cost and Sizing:
            - 2 cubic meter plant: ₹25,000 to ₹35,000
            - 4 cubic meter plant: ₹40,000 to ₹55,000
            - Government subsidies: 50-80% depending on state
            - Daily gas production: 1-2 cubic meters for average family
            """,
            
            "government_schemes.txt": """
            Government Schemes for Renewable Energy in Rural Areas
            
            Central Government Schemes:
            1. PM-KUSUM (Pradhan Mantri Kisan Urja Suraksha evam Utthaan Mahabhiyan)
               - Solar pumps for irrigation
               - Grid-connected solar power plants
               - Solarization of existing agricultural pumps
            
            2. Off-Grid Solar Programme
               - Solar lighting systems
               - Solar power plants for remote villages
               - Solar study lamps for students
            
            3. National Biogas and Manure Management Programme
               - Subsidies for biogas plants
               - Technical support and training
            
            State-Level Incentives:
            - Additional subsidies varying by state
            - Net metering policies
            - Tax exemptions
            - Accelerated depreciation benefits
            
            How to Apply:
            1. Contact local renewable energy development agency
            2. Submit application with required documents
            3. Technical feasibility assessment
            4. Approval and subsidy disbursement
            5. Installation and commissioning
            """,
            
            "maintenance_guide.txt": """
            Maintenance Guide for Renewable Energy Systems
            
            Solar Panel Maintenance:
            Daily:
            - Check for shadows or obstructions
            - Monitor energy production readings
            
            Monthly:
            - Clean panels with soft cloth and water
            - Check for physical damage or cracks
            - Inspect wiring connections
            
            Quarterly:
            - Check battery electrolyte levels (if applicable)
            - Inspect charge controller settings
            - Test system performance
            
            Annually:
            - Professional inspection and cleaning
            - Inverter maintenance
            - Battery replacement assessment
            
            Wind Turbine Maintenance:
            Weekly:
            - Visual inspection for damage
            - Check guy wires and anchors
            
            Monthly:
            - Lubricate moving parts
            - Check brake systems
            - Inspect electrical connections
            
            Biogas Plant Maintenance:
            Daily:
            - Feed organic waste regularly
            - Check gas pressure
            
            Weekly:
            - Remove slurry from outlet
            - Check for gas leaks
            
            Monthly:
            - Clean inlet and outlet pipes
            - Check pH levels of slurry
            """
        }
        
        # Create docs directory and files
        os.makedirs("docs", exist_ok=True)
        for filename, content in renewable_energy_data.items():
            filepath = os.path.join("docs", filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Created comprehensive dataset file: {filename}")
        
        return list(renewable_energy_data.keys())
    
    def load_documents(self) -> List[Dict]:
        """Load and process documents with comprehensive error handling"""
        documents = []
        docs_dir = "docs"
        
        # Create comprehensive dataset if no documents exist
        if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
            logger.info("Creating comprehensive renewable energy dataset...")
            self.create_comprehensive_dataset()
        
        # Load all documents
        for file_name in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file_name)
            text = ""
            
            try:
                if file_name.endswith(".pdf"):
                    with open(file_path, "rb") as file:
                        pdf = PdfReader(file)
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    logger.info(f"Loaded PDF: {file_name}")
                
                elif file_name.endswith((".txt", ".md")):
                    with open(file_path, "r", encoding="utf-8") as file:
                        text = file.read()
                    logger.info(f"Loaded text file: {file_name}")
                
                if text.strip():
                    documents.append({
                        "id": file_name.replace(".pdf", "").replace(".txt", "").replace(".md", ""),
                        "content": text,
                        "metadata": {
                            "source": file_name,
                            "timestamp": datetime.now().isoformat(),
                            "type": "renewable_energy_guide"
                        }
                    })
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def process_documents(self, documents: List[Dict]) -> None:
        """Process and store documents in vector database"""
        try:
            # Clear existing collection
            try:
                existing_ids = self.collection.get()['ids']
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
                    logger.info(f"Cleared {len(existing_ids)} existing documents")
            except Exception as e:
                logger.warning(f"Error clearing collection: {str(e)}")
            
            total_chunks = 0
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc["content"])
                
                if not chunks:
                    logger.warning(f"No chunks created for document {doc['id']}")
                    continue
                
                # Generate embeddings
                embeddings = self.embedding_model.embed_documents(chunks)
                
                # Prepare metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(chunks):
                    metadata = doc["metadata"].copy()
                    metadata.update({
                        "chunk_id": i,
                        "chunk_text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                    })
                    chunk_metadata.append(metadata)
                
                # Store in ChromaDB
                chunk_ids = [f"{doc['id']}_chunk_{i}" for i in range(len(chunks))]
                
                self.collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=chunk_metadata,
                    ids=chunk_ids
                )
                
                total_chunks += len(chunks)
                logger.info(f"Processed {doc['id']}: {len(chunks)} chunks")
            
            logger.info(f"Successfully processed {len(documents)} documents with {total_chunks} total chunks")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def query_rag(self, query: str, top_k: int = 3) -> Dict:
        """Enhanced RAG query with better response handling"""
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Retrieve relevant chunks
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return {
                    "answer": "I don't have specific information about that topic. Could you please ask about solar energy, wind power, biogas, government schemes, or maintenance?",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Combine retrieved context
            context_pieces = results["documents"][0]
            context = "\n\n".join(context_pieces)
            
            # Get source information
            sources = [meta["source"] for meta in results["metadatas"][0]] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else [1.0] * len(context_pieces)
            
            # Calculate confidence (inverse of average distance)
            avg_distance = sum(distances) / len(distances)
            confidence = max(0.0, 1.0 - avg_distance)
            
            # Create enhanced prompt
            prompt = self.create_enhanced_prompt(query, context)
            
            # Generate response
            response = self.llm(prompt)
            
            processing_time = time.time() - start_time
            
            # Store in chat history
            self.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "sources": list(set(sources)),
                "confidence": confidence,
                "processing_time": processing_time
            })
            
            return {
                "answer": response,
                "sources": list(set(sources)),
                "confidence": confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": 0.0
            }
    
    def create_enhanced_prompt(self, query: str, context: str) -> str:
        """Create an enhanced prompt for better responses"""
        return f"""You are a helpful AI assistant specializing in renewable energy solutions for rural communities in India. 
You provide practical, clear, and actionable information about solar, wind, biogas, and other renewable energy technologies.

Context Information:
{context}

User Question: {query}

Instructions:
1. Provide a clear, practical answer based on the context
2. Use simple language suitable for rural users
3. Include specific costs, benefits, and steps when relevant
4. Mention government schemes or subsidies if applicable
5. If the question is not about renewable energy, politely redirect to renewable energy topics
6. Keep the response concise but comprehensive

Answer:"""
    
    def listen_to_voice(self) -> Optional[str]:
        """Convert speech to text"""
        if not self.voice_enabled:
            return None
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                st.info("Listening... Please speak your question.")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Voice input recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return None
        except Exception as e:
            logger.error(f"Voice recognition error: {str(e)}")
            st.error(f"Voice recognition error: {str(e)}")
            return None
    
    def speak_response(self, text: str):
        """Convert text to speech"""
        if not self.voice_enabled:
            return
        
        try:
            def speak():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            # Run TTS in a separate thread to avoid blocking
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")

# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    """Initialize and return the chatbot instance"""
    try:
        chatbot = RenewableEnergyChatbot()
        documents = chatbot.load_documents()
        if documents:
            chatbot.process_documents(documents)
            return chatbot
        else:
            st.error("Failed to load documents. Please check the logs.")
            return None
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        logger.error(f"Chatbot initialization error: {str(e)}")
        return None

def apply_enhanced_css():
    """Apply enhanced CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .chat-container {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
        }
        
        .user-message {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #2196F3;
        }
        
        .bot-response {
            background: #f1f8e9;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #4CAF50;
        }
        
        .metrics-container {
            background: #fff3e0;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #ff9800;
        }
        
        .source-info {
            background: #e8f5e8;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.8rem;
            margin-top: 1rem;
            color: #2e7d32;
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #45a049, #4CAF50);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        .language-selector {
            background: #fff;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 0.5rem;
        }
        
        .voice-button {
            background: linear-gradient(45deg, #FF6B6B, #FF5252);
            border: none;
            border-radius: 50px;
            padding: 1rem;
            color: white;
            font-weight: bold;
            margin: 0.5rem;
        }
        
        .statistics-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Enhanced main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="🌱 Rural Renewable Energy Assistant",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply enhanced styling
    apply_enhanced_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 AI-Powered Renewable Energy Assistant</h1>
        <p>Empowering Rural Communities with Clean Energy Solutions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    if not chatbot:
        st.error("❌ Failed to initialize the chatbot. Please check the logs and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Language selector (placeholder for future multilingual support)
        language = st.selectbox(
            "Select Language",
            ["English", "Hindi", "Tamil", "Bengali"],
            index=0,
            key="language_select"
        )
        
        # Voice settings
        st.subheader("🎤 Voice Features")
        voice_enabled = st.checkbox("Enable Voice Input/Output", value=chatbot.voice_enabled)
        
        # Statistics
        st.subheader("📊 Usage Statistics")
        if chatbot.chat_history:
            st.metric("Total Queries", len(chatbot.chat_history))
            avg_confidence = sum(chat["confidence"] for chat in chatbot.chat_history) / len(chatbot.chat_history)
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
            avg_time = sum(chat["processing_time"] for chat in chatbot.chat_history) / len(chatbot.chat_history)
            st.metric("Avg. Response Time", f"{avg_time:.2f}s")
        
        # Quick topics
        st.subheader("📋 Quick Topics")
        quick_topics = [
            "Solar energy basics",
            "Wind power for rural areas",
            "Biogas installation guide",
            "Government subsidies",
            "Maintenance tips",
            "Cost calculation"
        ]
        
        for topic in quick_topics:
            if st.button(topic, key=f"quick_{topic}"):
                st.session_state.user_input = f"Tell me about {topic.lower()}"
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat input
        user_input = st.text_input(
            "Ask about renewable energy solutions:",
            placeholder="e.g., How much does a solar system cost for a rural home?",
            key="user_input"
        )
        
        # Voice input button
        if voice_enabled and chatbot.voice_enabled:
            if st.button("🎤 Voice Input", key="voice_button"):
                voice_text = chatbot.listen_to_voice()
                if voice_text:
                    st.session_state.user_input = voice_text
                    user_input = voice_text
    
    with col2:
        # Submit button
        submit_clicked = st.button("💬 Send", type="primary")
    
    # Process query
    if (submit_clicked or user_input) and user_input.strip():
        with st.spinner("🔍 Searching for the best answer..."):
            # Query the chatbot
            result = chatbot.query_rag(user_input)
            
            # Display user question
            st.markdown(f"""
            <div class="user-message">
                <strong>❓ Your Question:</strong><br>
                {user_input}
            </div>
            """, unsafe_allow_html=True)
            
            # Display bot response
            st.markdown(f"""
            <div class="bot-response">
                <strong>🤖 Answer:</strong><br>
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics and sources
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']:.2f}")
            with col2:
                st.metric("Response Time", f"{result['processing_time']:.2f}s")
            with col3:
                st.metric("Sources", len(result['sources']))
            
            # Display sources
            if result['sources']:
                st.markdown(f"""
                <div class="source-info">
                    <strong>📚 Sources:</strong> {', '.join(result['sources'])}
                </div>
                """, unsafe_allow_html=True)
            
            # Voice output
            if voice_enabled and chatbot.voice_enabled:
                if st.button("🔊 Listen to Answer"):
                    chatbot.speak_response(result['answer'])
        
        # Clear input
        st.session_state.user_input = ""
    
    # Chat history
    if chatbot.chat_history:
        st.subheader("💭 Recent Conversations")
        
        # Show last 5 conversations
        for i, chat in enumerate(reversed(chatbot.chat_history[-5:])):
            with st.expander(f"Q: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Q: {chat['query']}"):
                st.write(f"**Answer:** {chat['response']}")
                st.write(f"**Confidence:** {chat['confidence']:.2f}")
                st.write(f"**Sources:** {', '.join(chat['sources']) if chat['sources'] else 'None'}")
                st.write(f"**Time:** {chat['timestamp']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🌱 Renewable Energy Chatbot v1.0 | Empowering Rural Communities with Clean Energy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")
