# Enhanced AI-powered Renewable Energy Chatbot for Rural Communities
# 75% Implementation with News Page, Fixed UI, and Modular Structure

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import requests
from urllib.parse import urlparse
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
                # Enhanced rule-based responses for demo purposes
                responses = {
                    "solar": "Solar energy harnesses sunlight using photovoltaic panels. It's ideal for rural areas due to low maintenance requirements and scalability. A 1kW system costs ₹40,000-80,000. Government subsidies can reduce costs by 30-70%. The system has a 25+ year lifespan with 4-6 year payback period.",
                    "wind": "Wind energy uses turbines to convert wind into electricity. Small-scale turbines (1-10kW) cost ₹1-8 lakhs and are suitable for rural communities with consistent wind speeds above 3 m/s. They require open areas and regular maintenance.",
                    "biogas": "Biogas is produced from organic waste through anaerobic digestion. A 2 cubic meter plant costs ₹25,000-35,000 and can provide cooking gas for a family of 4-5 people. It also produces high-quality organic fertilizer.",
                    "maintenance": "Regular maintenance includes: Solar - monthly cleaning, quarterly connection checks, annual professional inspection. Wind - weekly visual inspection, monthly lubrication, quarterly brake system checks.",
                    "cost": "Typical costs: Solar 1kW system: ₹40,000-80,000, Wind 1kW turbine: ₹1-2 lakhs, Biogas 2m³ plant: ₹25,000-35,000. Government subsidies available: 30-70% for solar, 50-80% for biogas.",
                    "installation": "Installation process: 1) Site assessment and feasibility study, 2) System design and component selection, 3) Permits and government approvals, 4) Professional installation, 5) Grid connection and commissioning, 6) Performance testing.",
                    "subsidy": "Government schemes: PM-KUSUM for solar pumps, Off-grid Solar Programme, National Biogas Programme. Central subsidies: 30-70% for residential solar, 50-80% for biogas. Apply through local renewable energy development agencies."
                }
                
                prompt_lower = prompt.lower()
                for key, response in responses.items():
                    if key in prompt_lower:
                        return f"Based on your query about renewable energy: {response}"
                
                return "I can help you with information about solar energy, wind power, biogas, maintenance, costs, subsidies, and installation. Please ask a specific question about renewable energy solutions for rural areas."
        
        logger.info("Using fallback LLM for demonstration")
        return FallbackLLM()
    
    def create_comprehensive_dataset(self):
        """Create a comprehensive renewable energy dataset"""
        renewable_energy_data = {
            "solar_energy_guide.txt": """
            Comprehensive Solar Energy Guide for Rural Communities
            
            Solar energy is the most accessible renewable energy source for rural areas in India. 
            Solar panels convert sunlight directly into electricity using photovoltaic cells.
            
            Benefits for Rural Areas:
            - Minimal maintenance requirements - just monthly cleaning
            - Highly scalable from 1kW home systems to 100kW community installations
            - Reduces electricity bills by 70-90%
            - Perfect for remote locations without grid connectivity
            - Can power water pumps, LED lighting, fans, and small appliances
            - 25+ year lifespan with 80% efficiency retention
            
            Types of Solar Systems:
            1. On-grid systems: Connected to electricity grid, excess power sold back
            2. Off-grid systems: Independent with battery backup for 24/7 power
            3. Hybrid systems: Combination providing grid backup and independence
            
            Detailed Cost Breakdown (2024 rates):
            - 1kW system: ₹45,000-65,000 (powers 4-5 LED lights, 2 fans, TV)
            - 3kW system: ₹1,20,000-1,80,000 (average rural household needs)
            - 5kW system: ₹2,00,000-3,00,000 (larger homes with appliances)
            - 10kW system: ₹3,50,000-5,00,000 (small businesses, community centers)
            
            Government Subsidies Available:
            - Central Government: 40% subsidy up to 3kW, 20% for 3-10kW
            - State subsidies: Additional 10-30% depending on state
            - PM-KUSUM scheme: 60% subsidy for solar water pumps
            - Net metering: Sell excess power back to grid
            
            Installation Process:
            1. Site survey and energy audit (1-2 days)
            2. System design and component selection (2-3 days)
            3. Government approvals and net metering application (15-30 days)
            4. Procurement and installation (3-5 days)
            5. Grid connection and commissioning (5-10 days)
            6. Subsidy claim processing (30-90 days)
            
            Maintenance Schedule:
            Daily: Monitor energy generation through app
            Weekly: Visual inspection for damage or shading
            Monthly: Panel cleaning with soft cloth and water
            Quarterly: Electrical connection inspection
            Annually: Professional maintenance and performance check
            """,
            
            "wind_energy_rural.txt": """
            Wind Energy Solutions for Rural Communities
            
            Wind energy is suitable for rural areas with consistent wind speeds above 4 m/s.
            Small wind turbines convert kinetic wind energy into electricity.
            
            Wind Resource Assessment:
            - Minimum wind speed: 3 m/s for small turbines
            - Optimal wind speed: 5-15 m/s for maximum efficiency
            - Site requirements: Open area with 150m radius clearance
            - Height advantage: Every 10m height increases wind speed by 10-20%
            
            Small Wind Turbine Categories:
            - Micro turbines (under 1kW): ₹60,000-1,20,000
              * Suitable for: LED lighting, mobile charging, small appliances
              * Power output: 2-5 kWh per day in good wind conditions
            
            - Mini turbines (1-5kW): ₹1,50,000-6,00,000
              * Suitable for: Rural homes, small farms, community buildings
              * Power output: 5-25 kWh per day depending on wind
            
            - Small turbines (5-25kW): ₹6,00,000-25,00,000
              * Suitable for: Small businesses, village clusters, agricultural processing
              * Power output: 25-125 kWh per day in favorable conditions
            
            Installation Requirements:
            - Foundation: Concrete foundation 2-3m deep
            - Tower height: 9-30m depending on turbine size
            - Grid connection: Inverter and safety systems required
            - Permits: State electricity board approval needed
            
            Maintenance Requirements:
            Weekly: Visual inspection for damage, loose components
            Monthly: Lubrication of bearings and moving parts
            Quarterly: Brake system inspection and testing
            Bi-annually: Professional electrical system check
            Annually: Complete turbine inspection and service
            
            Economic Analysis:
            - Payback period: 6-10 years depending on wind resource
            - Operating costs: ₹0.50-1.50 per kWh
            - Lifespan: 20-25 years with proper maintenance
            - Capacity factor: 15-35% in Indian conditions
            """,
            
            "biogas_comprehensive.txt": """
            Biogas Technology Comprehensive Guide
            
            Biogas is produced through anaerobic digestion of organic matter.
            Perfect for rural families with livestock and kitchen waste.
            
            Biogas Composition and Benefits:
            - Methane (50-70%): Primary combustible gas
            - Carbon dioxide (30-40%): Inert gas
            - Hydrogen sulfide (<2%): Removed through purification
            - Heating value: 20-25 MJ per cubic meter
            
            Raw Materials for Biogas:
            - Fresh cow dung: 1 kg produces 0.3-0.4 cubic meters gas
            - Buffalo dung: 1 kg produces 0.25-0.35 cubic meters gas
            - Kitchen waste: 1 kg produces 0.4-0.6 cubic meters gas
            - Agricultural residue: 1 kg produces 0.2-0.4 cubic meters gas
            - Poultry waste: 1 kg produces 0.5-0.8 cubic meters gas
            
            Plant Size and Family Requirements:
            - 1 cubic meter plant: 2-3 person family, 1-2 cattle
              * Gas production: 0.5-1 cubic meters per day
              * Cooking time: 1-2 hours daily
              * Cost: ₹15,000-25,000
            
            - 2 cubic meter plant: 4-5 person family, 3-4 cattle
              * Gas production: 1-2 cubic meters per day
              * Cooking time: 2-4 hours daily
              * Cost: ₹25,000-40,000
            
            - 4 cubic meter plant: 6-8 person family, 6-8 cattle
              * Gas production: 2-4 cubic meters per day
              * Cooking time: 4-8 hours daily
              * Cost: ₹45,000-70,000
            
            Types of Biogas Plants:
            1. Fixed Dome (Chinese) Model:
               - Underground construction, durable
               - 15-20 year lifespan
               - Higher initial cost, lower maintenance
            
            2. Floating Gas Holder (Indian) Model:
               - Above ground gas storage
               - Easy maintenance and operation
               - 10-15 year lifespan
            
            3. Bag Type (Balloon) Model:
               - Portable, low cost option
               - 3-5 year lifespan
               - Suitable for small families
            
            Government Support:
            - National Biogas Programme: 50-80% subsidy
            - Additional state subsidies: 10-20%
            - Technical support and training provided
            - Loan facilities available through banks
            
            Economic Benefits:
            - LPG savings: ₹200-500 per month per cubic meter capacity
            - Fertilizer value: ₹100-200 per month from slurry
            - Payback period: 2-4 years with subsidies
            - Carbon credits: Additional income potential
            """,
            
            "government_schemes_2024.txt": """
            Government Renewable Energy Schemes 2024 Update
            
            Central Government Schemes:
            
            1. PM-KUSUM (Pradhan Mantri Kisan Urja Suraksha evam Utthaan Mahabhiyan):
            Component A: Solar power plants (500kW to 2MW)
               - 30% central subsidy, 30% state subsidy, 40% farmer contribution
               - Farmers can sell power to grid and earn ₹1 lakh per MW annually
            
            Component B: Standalone solar water pumps
               - 60% subsidy (30% central + 30% state), 40% farmer contribution
               - 7.5HP pumps subsidized up to ₹3 lakh
            
            Component C: Grid-connected solar pump solarization
               - 60% subsidy for existing pump solarization
               - Reduces electricity bill by 95%
            
            2. Grid Connected Rooftop Solar Programme Phase-II:
               - Residential: 40% subsidy up to 3kW, 20% for 3-10kW
               - Maximum subsidy: ₹78,000 for 10kW system
               - Net metering facility for excess power sale
            
            3. Off-Grid and Decentralized Solar Applications:
               - Solar street lights: 70% subsidy
               - Solar study lamps: 50% subsidy
               - Solar power plants for villages: 60% subsidy
            
            4. National Biogas and Manure Management Programme:
               - Family size biogas plants: 50-80% subsidy
               - Community biogas plants: 40-60% subsidy
               - Biogas stoves and lamps: 50% subsidy
            
            State-Specific Schemes (Major States):
            
            Gujarat:
               - Additional 20% state subsidy on solar
               - Surya Gujarat Yojana: Interest-free loans
               - Wind energy: Accelerated depreciation benefits
            
            Rajasthan:
               - Solar energy policy: Additional incentives for desert areas
               - Wind energy zones with transmission support
               - Biomass policy: Support for agricultural residue
            
            Tamil Nadu:
               - Wind energy leader: Simplified approval process
               - Solar rooftop mission: State-specific incentives
               - Biogas promotion in rural areas
            
            Maharashtra:
               - Agri-voltaics: Solar on agricultural land
               - Wind-solar hybrid policy
               - Sugar mill bagasse-based power incentives
            
            Application Process:
            1. Visit local Renewable Energy Development Agency (REDA)
            2. Submit application with required documents:
               - Identity proof, address proof
               - Land ownership documents
               - Electricity connection details
               - Bank account details
            3. Technical feasibility assessment
            4. Approval and work order issuance
            5. Installation and commissioning
            6. Subsidy disbursement (30-90 days)
            
            New Initiatives 2024:
            - Green Hydrogen Mission: ₹19,744 crore allocation
            - Battery storage incentives: 40% cost reduction target
            - Electric vehicle charging from renewables: Priority support
            - Carbon credit mechanism: Additional revenue for farmers
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

class NewsService:
    """Service to fetch renewable energy news"""
    
    def __init__(self):
        self.cache_duration = 3600  # 1 hour cache
        self.last_fetch = None
        self.cached_news = []
    
    def get_renewable_energy_news(self) -> List[Dict]:
        """Get renewable energy news from multiple sources"""
        # For demo purposes, return static news. In production, integrate with news APIs
        current_time = datetime.now()
        
        # Check cache
        if (self.last_fetch and 
            (current_time - self.last_fetch).seconds < self.cache_duration and 
            self.cached_news):
            return self.cached_news
        
        # Simulated news data (in production, replace with actual API calls)
        news_data = [
            {
                "title": "India Launches New Solar Rooftop Scheme with Enhanced Subsidies",
                "summary": "Government announces increased subsidies for residential solar installations, up to 70% for systems under 3kW capacity. The scheme aims to reach 40 GW rooftop solar by 2025.",
                "source": "Ministry of New and Renewable Energy",
                "url": "https://mnre.gov.in/solar-rooftop-scheme",
                "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                "category": "Government Policy",
                "tags": ["Solar", "Subsidies", "Policy", "India"]
            },
            {
                "title": "Rural Wind Energy Projects Show 25% Growth in 2024",
                "summary": "Small-scale wind installations in rural areas increased significantly, driven by improved technology and financing options. Average project size: 5-50kW suitable for villages.",
                "source": "Indian Wind Power Association",
                "url": "https://indianwindpower.com/rural-growth",
                "timestamp": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
                "category": "Market Update",
                "tags": ["Wind", "Rural", "Growth", "Statistics"]
            },
            {
                "title": "Biogas Plants Reduce Rural Energy Costs by 60% - New Study",
                "summary": "Research shows family biogas plants save ₹3,000-5,000 monthly on cooking fuel. Government targets 1 million new installations in next two years.",
                "source": "National Institute of Rural Development",
                "url": "https://nird.gov.in/biogas-study-2024",
                "timestamp": (datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
                "category": "Research",
                "tags": ["Biogas", "Cost Savings", "Rural", "Study"]
            },
            {
                "title": "PM-KUSUM Scheme Crosses 2 Lakh Solar Pump Installations",
                "summary": "Landmark achievement in agricultural solar pumps under PM-KUSUM. Farmers report 70% reduction in irrigation costs and improved crop yields.",
                "source": "PIB India",
                "url": "https://pib.gov.in/kusum-milestone",
                "timestamp": (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
                "category": "Achievement",
                "tags": ["PM-KUSUM", "Solar Pumps", "Agriculture", "Milestone"]
            },
            {
                "title": "International Solar Alliance Announces $1B Fund for Rural Projects",
                "summary": "New funding mechanism specifically targets small-scale renewable projects in developing countries. Focus on off-grid solutions and micro-grids.",
                "source": "International Solar Alliance",
                "url": "https://isolaralliance.org/rural-fund",
                "timestamp": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                "category": "International",
                "tags": ["ISA", "Funding", "International", "Rural"]
            },
            {
                "title": "New Battery Storage Technology Reduces Solar System Costs by 30%",
                "summary": "Indian startup develops affordable lithium-ion battery alternative using local materials. Expected to make solar-battery systems accessible to rural households.",
                "source": "Clean Energy Review",
                "url": "https://cleanenergyreview.in/battery-breakthrough",
                "timestamp": (datetime.now() - timedelta(days=1, hours=6)).strftime("%Y-%m-%d %H:%M"),
                "category": "Technology",
                "tags": ["Battery", "Innovation", "Solar", "Cost Reduction"]
            },
            {
                "title": "Rajasthan Leads in Agri-Voltaics: Solar Panels Over Farmland",
                "summary": "State government promotes dual-use of agricultural land for crop production and solar power generation. Pilots show 15% increase in crop yields under solar panels.",
                "source": "Rajasthan Renewable Energy Corporation",
                "url": "https://rrecl.com/agri-voltaics",
                "timestamp": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
                "category": "Innovation",
                "tags": ["Agri-voltaics", "Rajasthan", "Farming", "Solar"]
            },
            {
                "title": "Global Renewable Energy Investment Hits Record $1.8 Trillion",
                "summary": "IEA reports unprecedented investment in renewable energy globally. Solar and wind dominate with 85% share. Rural electrification projects receive significant funding.",
                "source": "International Energy Agency",
                "url": "https://iea.org/renewable-investment-2024",
                "timestamp": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
                "category": "Global",
                "tags": ["Investment", "Global", "IEA", "Statistics"]
            }
        ]
        
        self.cached_news = news_data
        self.last_fetch = current_time
        return news_data
    
    def get_news_by_category(self, category: str) -> List[Dict]:
        """Filter news by category"""
        all_news = self.get_renewable_energy_news()
        if category == "All":
            return all_news
        return [news for news in all_news if news["category"] == category]

# Initialize services
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

@st.cache_resource
def initialize_news_service():
    """Initialize news service"""
    return NewsService()

def apply_enhanced_css():
    """Apply enhanced CSS styling with fixed dark theme"""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --bg-primary: #0f1419;
            --bg-secondary: #1a1f2e;
            --bg-tertiary: #242937;
            --text-primary: #ffffff;
            --text-secondary: #b3b3b3;
            --accent-green: #4CAF50;
            --accent-blue: #2196F3;
            --border-color: #333;
            --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* Global Streamlit overrides */
        .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: var(--card-shadow);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Fixed response box styling */
        .response-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--card-shadow);
        }
        
        .user-message {
            background: var(--bg-tertiary);
            border-left: 4px solid var(--accent-blue);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }
        
        .bot-response {
            background: var(--bg-secondary);
            border-left: 4px solid var(--accent-green);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        /* News card styling */
        .news-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            box-shadow: var(--card-shadow);
        }
        
        .news-card:hover {
            border-color: var(--accent-green);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        }
        
        .news-title {
            color: var(--text-primary);
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            line-height: 1.4;
        }
        
        .news-summary {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }
        
        .news-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .news-category {
            background: var(--accent-green);
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-weight: 500;
        }
        
        .news-timestamp {
            color: var(--text-secondary);
        }
        
        .news-link {
            color: var(--accent-blue);
            text-decoration: none;
            font-weight: 500;
        }
        
        .news-link:hover {
            text-decoration: underline;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, var(--accent-green), #45a049);
            color: white;
            border: none;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #45a049, var(--accent-green));
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        /* Metrics styling */
        .metric-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-green);
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }
        
        /* Source info styling */
        .source-info {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            padding: 0.8rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-top: 1rem;
            color: var(--text-secondary);
        }
        
        /* Loading spinner */
        .loading-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .spinner {
            border: 3px solid var(--bg-tertiary);
            border-top: 3px solid var(--accent-green);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin-right: 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Input field styling */
        .stTextInput > div > div > input {
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: 'Inter', sans-serif;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-green);
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
        }
        
        /* Quick topics styling */
        .quick-topic-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.3rem 0;
            width: 100%;
            text-align: left;
            color: var(--text-primary);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .quick-topic-btn:hover {
            border-color: var(--accent-green);
            background: var(--bg-secondary);
        }
        
        /* Filter buttons */
        .filter-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin: 0.2rem;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }
        
        .filter-btn.active {
            background: var(--accent-green);
            border-color: var(--accent-green);
            color: white;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-green);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .news-card {
                padding: 1rem;
            }
            
            .news-meta {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_loading_spinner(text="Processing your request..."):
    """Create a loading spinner with text"""
    return f"""
    <div class="loading-container">
        <div class="spinner"></div>
        <span>{text}</span>
    </div>
    """

def main_qa_page(chatbot):
    """Main Q&A page"""
    st.markdown("""
    <div class="main-header">
        <h1>🌱 AI-Powered Renewable Energy Assistant</h1>
        <p>Empowering Rural Communities with Clean Energy Solutions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use unique key and avoid session state conflicts
        user_query = st.text_input(
            "Ask about renewable energy solutions:",
            placeholder="e.g., How much does a solar system cost for a rural home?",
            key="main_query_input"
        )
    
    with col2:
        submit_clicked = st.button("💬 Send", type="primary", key="main_submit")
    
    # Process query
    if submit_clicked and user_query.strip():
        # Show loading spinner
        loading_placeholder = st.empty()
        loading_placeholder.markdown(create_loading_spinner("🔍 Searching for the best answer..."), unsafe_allow_html=True)
        
        # Query the chatbot
        result = chatbot.query_rag(user_query)
        
        # Clear loading spinner
        loading_placeholder.empty()
        
        # Display user question
        st.markdown(f"""
        <div class="user-message">
            <strong>❓ Your Question:</strong><br>
            {user_query}
        </div>
        """, unsafe_allow_html=True)
        
        # Display bot response with fixed styling
        st.markdown(f"""
        <div class="bot-response">
            <strong>🤖 Answer:</strong><br>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{result['confidence']:.0%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{result['processing_time']:.1f}s</div>
                <div class="metric-label">Response Time</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(result['sources'])}</div>
                <div class="metric-label">Sources</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display sources
        if result['sources']:
            st.markdown(f"""
            <div class="source-info">
                <strong>📚 Sources:</strong> {', '.join(result['sources'])}
            </div>
            """, unsafe_allow_html=True)

def news_page(news_service):
    """News page with renewable energy updates"""
    st.markdown("""
    <div class="main-header">
        <h1>📰 Renewable Energy News & Updates</h1>
        <p>Latest developments in clean energy sector</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Category filter
    categories = ["All", "Government Policy", "Market Update", "Research", "Achievement", 
                  "International", "Technology", "Innovation", "Global"]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Filter by Category:")
        
    # Create filter buttons
    cols = st.columns(len(categories))
    selected_category = "All"
    
    for i, category in enumerate(categories):
        with cols[i]:
            if st.button(category, key=f"cat_{category}"):
                selected_category = category
    
    # Get news based on selection
    if 'selected_news_category' not in st.session_state:
        st.session_state.selected_news_category = "All"
    
    # Update selection if button was clicked
    for category in categories:
        if f"cat_{category}" in st.session_state and st.session_state[f"cat_{category}"]:
            st.session_state.selected_news_category = category
            break
    
    news_items = news_service.get_news_by_category(st.session_state.selected_news_category)
    
    # Display news items
    st.markdown(f"### Latest News ({len(news_items)} articles)")
    
    for news_item in news_items:
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{news_item['title']}</div>
            <div class="news-summary">{news_item['summary']}</div>
            <div class="news-meta">
                <span class="news-category">{news_item['category']}</span>
                <span class="news-timestamp">📅 {news_item['timestamp']}</span>
                <a href="{news_item['url']}" target="_blank" class="news-link">🔗 Read Full Article</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

def cost_calculator_page():
    """Cost calculator page for renewable energy systems"""
    st.markdown("""
    <div class="main-header">
        <h1>💰 Renewable Energy Cost Calculator</h1>
        <p>Calculate costs and savings for your renewable energy system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System selection
    system_type = st.selectbox(
        "Select Renewable Energy System:",
        ["Solar PV System", "Wind Turbine", "Biogas Plant", "Solar Water Heater"],
        key="calc_system_type"
    )
    
    if system_type == "Solar PV System":
        capacity = st.slider("System Capacity (kW)", 1, 10, 3, key="solar_capacity")
        
        # Cost calculation
        base_cost_per_kw = 50000  # ₹50,000 per kW
        total_cost = capacity * base_cost_per_kw
        subsidy_rate = 0.4 if capacity <= 3 else 0.2
        subsidy_amount = total_cost * subsidy_rate
        final_cost = total_cost - subsidy_amount
        
        # Monthly savings
        monthly_generation = capacity * 4 * 30  # 4 hours avg daily
        monthly_savings = monthly_generation * 4  # ₹4 per kWh
        payback_years = final_cost / (monthly_savings * 12)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="response-container">
                <h3>💵 Cost Breakdown</h3>
                <p><strong>System Cost:</strong> ₹{total_cost:,}</p>
                <p><strong>Government Subsidy ({subsidy_rate:.0%}):</strong> -₹{subsidy_amount:,}</p>
                <p><strong>Your Investment:</strong> ₹{final_cost:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="response-container">
                <h3>💡 Savings & Returns</h3>
                <p><strong>Monthly Generation:</strong> {monthly_generation:.0f} kWh</p>
                <p><strong>Monthly Savings:</strong> ₹{monthly_savings:.0f}</p>
                <p><strong>Payback Period:</strong> {payback_years:.1f} years</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif system_type == "Biogas Plant":
        family_size = st.slider("Family Size", 2, 10, 5, key="biogas_family")
        cattle_count = st.slider("Number of Cattle", 1, 10, 3, key="biogas_cattle")
        
        # Calculate plant size needed
        plant_size = max(2, (family_size + cattle_count) // 2)
        base_cost = plant_size * 15000  # ₹15,000 per cubic meter
        subsidy_amount = base_cost * 0.6  # 60% subsidy
        final_cost = base_cost - subsidy_amount
        
        # Monthly savings
        gas_production = plant_size * 0.8  # cubic meters per day
        monthly_gas = gas_production * 30
        lpg_cylinder_equivalent = monthly_gas / 14.2  # 14.2 cubic meters per cylinder
        monthly_savings = lpg_cylinder_equivalent * 900  # ₹900 per cylinder
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="response-container">
                <h3>🏭 System Details</h3>
                <p><strong>Recommended Size:</strong> {plant_size} cubic meters</p>
                <p><strong>Total Cost:</strong> ₹{base_cost:,}</p>
                <p><strong>Subsidy (60%):</strong> -₹{subsidy_amount:,}</p>
                <p><strong>Your Investment:</strong> ₹{final_cost:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="response-container">
                <h3>⚡ Production & Savings</h3>
                <p><strong>Daily Gas Production:</strong> {gas_production:.1f} cubic meters</p>
                <p><strong>LPG Cylinders Saved/Month:</strong> {lpg_cylinder_equivalent:.1f}</p>
                <p><strong>Monthly Savings:</strong> ₹{monthly_savings:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

def maintenance_guide_page():
    """Maintenance guide page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Maintenance Guide</h1>
        <p>Keep your renewable energy systems running efficiently</p>
    </div>
    """, unsafe_allow_html=True)
    
    system_type = st.selectbox(
        "Select System Type:",
        ["Solar PV System", "Wind Turbine", "Biogas Plant"],
        key="maint_system_type"
    )
    
    maintenance_guides = {
        "Solar PV System": {
            "Daily": [
                "Monitor energy production through app or display",
                "Check for any visible damage or shading on panels",
                "Ensure inverter status lights are normal (green)"
            ],
            "Monthly": [
                "Clean panels with soft cloth and clean water",
                "Check electrical connections for corrosion",
                "Inspect mounting structure for loose bolts",
                "Trim vegetation that might cause shading"
            ],
            "Quarterly": [
                "Check battery electrolyte levels (if applicable)",
                "Inspect charge controller settings",
                "Test system isolation switches",
                "Check earthing connections"
            ],
            "Annually": [
                "Professional system inspection and cleaning",
                "Inverter maintenance and firmware updates",
                "Battery performance assessment",
                "Electrical safety testing"
            ]
        },
        "Wind Turbine": {
            "Weekly": [
                "Visual inspection for damage or unusual noise",
                "Check guy wires and anchor points",
                "Verify turbine orientation and rotation"
            ],
            "Monthly": [
                "Lubricate bearings and moving parts",
                "Check brake system operation",
                "Inspect electrical connections",
                "Clean turbine blades if needed"
            ],
            "Quarterly": [
                "Professional tower and foundation inspection",
                "Electrical system testing",
                "Performance analysis and optimization"
            ],
            "Annually": [
                "Complete turbine overhaul",
                "Replace worn components",
                "Safety system testing",
                "Professional maintenance service"
            ]
        },
        "Biogas Plant": {
            "Daily": [
                "Feed organic waste according to schedule",
                "Check gas pressure gauge",
                "Monitor gas flame quality while cooking"
            ],
            "Weekly": [
                "Remove slurry from outlet chamber",
                "Check for gas leaks using soap solution",
                "Clean gas pipes and connections"
            ],
            "Monthly": [
                "Clean inlet and outlet pipes",
                "Check pH level of slurry (should be 6.8-7.2)",
                "Inspect gas holder for damage",
                "Add water if slurry becomes too thick"
            ],
            "Annually": [
                "Empty and clean entire plant",
                "Repair any cracks in structure",
                "Replace gas pipes if needed",
                "Professional inspection and servicing"
            ]
        }
    }
    
    if system_type in maintenance_guides:
        guide = maintenance_guides[system_type]
        
        for period, tasks in guide.items():
            st.markdown(f"""
            <div class="response-container">
                <h3>{period} Maintenance</h3>
                {"".join([f"<p>• {task}</p>" for task in tasks])}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Enhanced main Streamlit application with modular pages"""
    # Page configuration
    st.set_page_config(
        page_title="🌱 Rural Renewable Energy Assistant",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply enhanced styling
    apply_enhanced_css()
    
    # Initialize services
    chatbot = initialize_chatbot()
    news_service = initialize_news_service()
    
    if not chatbot:
        st.error("❌ Failed to initialize the chatbot. Please check the logs and try again.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🔧 Navigation")
        
        # Page selection
        page = st.selectbox(
            "Select Page",
            ["🏠 Main Q&A", "📰 News & Updates", "💰 Cost Calculator", "🔧 Maintenance Guide"],
            key="page_selector"
        )
        
        # Language selector
        st.subheader("🌐 Language")
        language = st.selectbox(
            "Select Language",
            ["English", "हिंदी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)"],
            index=0,
            key="language_select"
        )
        
        # Quick topics for main page
        if page == "🏠 Main Q&A":
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
                if st.button(topic, key=f"quick_{topic.replace(' ', '_')}"):
                    # Set session state to trigger query
                    st.session_state.quick_query = f"Tell me about {topic.lower()}"
        
        # Usage statistics
        st.subheader("📊 Usage Statistics")
        if chatbot.chat_history:
            st.metric("Total Queries", len(chatbot.chat_history))
            avg_confidence = sum(chat["confidence"] for chat in chatbot.chat_history) / len(chatbot.chat_history)
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
            avg_time = sum(chat["processing_time"] for chat in chatbot.chat_history) / len(chatbot.chat_history)
            st.metric("Avg. Response Time", f"{avg_time:.2f}s")
        else:
            st.info("No queries yet. Start asking questions!")
    
    # Main content area
    if page == "🏠 Main Q&A":
        main_qa_page(chatbot)
    elif page == "📰 News & Updates":
        news_page(news_service)
    elif page == "💰 Cost Calculator":
        cost_calculator_page()
    elif page == "🔧 Maintenance Guide":
        maintenance_guide_page()
    
    # Handle quick query from sidebar
    if 'quick_query' in st.session_state and page == "🏠 Main Q&A":
        quick_query = st.session_state.quick_query
        del st.session_state.quick_query
        
        # Show loading spinner
        loading_placeholder = st.empty()
        loading_placeholder.markdown(create_loading_spinner("🔍 Processing your quick question..."), unsafe_allow_html=True)
        
        # Process the query
        result = chatbot.query_rag(quick_query)
        
        # Clear loading spinner
        loading_placeholder.empty()
        
        # Display results
        st.markdown(f"""
        <div class="user-message">
            <strong>❓ Quick Question:</strong><br>
            {quick_query}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="bot-response">
            <strong>🤖 Answer:</strong><br>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat history section (only on main page)
    if page == "🏠 Main Q&A" and chatbot.chat_history:
        st.markdown("---")
        st.subheader("💭 Recent Conversations")
        
        # Show last 3 conversations
        for i, chat in enumerate(reversed(chatbot.chat_history[-3:])):
            with st.expander(f"Q: {chat['query'][:50]}..." if len(chat['query']) > 50 else f"Q: {chat['query']}"):
                st.markdown(f"""
                <div class="response-container">
                    <p><strong>Answer:</strong> {chat['response']}</p>
                    <p><strong>Confidence:</strong> {chat['confidence']:.2f}</p>
                    <p><strong>Sources:</strong> {', '.join(chat['sources']) if chat['sources'] else 'None'}</p>
                    <p><strong>Time:</strong> {chat['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 1rem; font-size: 0.9rem;">
        🌱 Renewable Energy Assistant v2.0 | Empowering Rural Communities with Clean Energy | 75% Implementation Complete
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")