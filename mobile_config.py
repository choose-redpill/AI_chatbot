# mobile_app.py - Enhanced mobile-friendly version of the chatbot

import streamlit as st
from chatbot import RenewableEnergyChatbot, initialize_chatbot
import json

def apply_mobile_css():
    """Apply mobile-optimized CSS"""
    st.markdown("""
    <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .main-header {
                padding: 1rem;
                font-size: 1.2rem;
            }
            
            .stButton > button {
                width: 100%;
                padding: 0.75rem;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            }
            
            .voice-button {
                width: 100%;
                padding: 1rem;
                font-size: 1.2rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            
            .chat-input {
                font-size: 1.1rem;
                padding: 1rem;
            }
            
            .bot-response {
                font-size: 1rem;
                line-height: 1.6;
            }
            
            .sidebar .sidebar-content {
                padding: 0.5rem;
            }
        }
        
        /* Touch-friendly interface */
        .touch-button {
            min-height: 44px;
            min-width: 44px;
            padding: 12px 16px;
            font-size: 16px;
            border-radius: 8px;
            margin: 8px 4px;
            touch-action: manipulation;
        }
        
        /* Improved readability on mobile */
        .mobile-text {
            font-size: 16px;
            line-height: 1.5;
            word-spacing: 0.1em;
        }
        
        /* PWA-ready styling */
        .pwa-header {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1rem;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Offline indicator */
        .offline-indicator {
            background: #ff9800;
            color: white;
            padding: 0.5rem;
            text-align: center;
            font-weight: bold;
        }
        
        /* Quick action buttons */
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 1rem 0;
        }
        
        .quick-action-btn {
            background: #E8F5E8;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .quick-action-btn:hover {
            background: #4CAF50;
            color: white;
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)

def create_pwa_manifest():
    """Create Progressive Web App manifest"""
    manifest = {
        "name": "Rural Renewable Energy Assistant",
        "short_name": "RenewableBot",
        "description": "AI-powered assistant for renewable energy in rural communities",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#4CAF50",
        "orientation": "portrait",
        "icons": [
            {
                "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='50' fill='%234CAF50'/%3E%3Ctext x='50' y='60' text-anchor='middle' fill='white' font-size='40'%3E🌱%3C/text%3E%3C/svg%3E",
                "sizes": "192x192",
                "type": "image/svg+xml"
            }
        ]
    }
    
    return json.dumps(manifest, indent=2)

def mobile_voice_interface(chatbot):
    """Enhanced mobile voice interface"""
    st.markdown("### 🎤 Voice Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", key="start_voice", help="Tap to start voice input"):
            if chatbot and chatbot.voice_enabled:
                with st.spinner("🔊 Listening..."):
                    voice_text = chatbot.listen_to_voice()
                    if voice_text:
                        st.session_state.mobile_voice_input = voice_text
                        st.success(f"📝 Heard: {voice_text}")
            else:
                st.error("Voice not available. Please check microphone permissions.")
    
    with col2:
        if st.button("🔊 Read Response", key="speak_response", help="Tap to hear the last answer"):
            if 'last_response' in st.session_state and chatbot and chatbot.voice_enabled:
                chatbot.speak_response(st.session_state.last_response)
                st.success("🔊 Playing audio...")

def quick_action_buttons():
    """Create quick action buttons for mobile"""
    st.markdown("### ⚡ Quick Questions")
    
    quick_questions = {
        "💰 Solar Costs": "How much does a solar system cost for a small rural home?",
        "🔧 Maintenance": "What maintenance is needed for solar panels?",
        "💨 Wind Energy": "Is wind energy suitable for my rural area?",
        "🏛️ Subsidies": "What government subsidies are available for renewable energy?",
        "⚡ Biogas": "How do I set up a biogas plant at home?",
        "📊 Savings": "How much money can I save with solar energy?"
    }
    
    # Create responsive grid
    cols = st.columns(2)
    for i, (button_text, question) in enumerate(quick_questions.items()):
        with cols[i % 2]:
            if st.button(button_text, key=f"quick_{i}", help=f"Ask: {question}"):
                st.session_state.mobile_quick_input = question

def mobile_chat_interface(chatbot):
    """Mobile-optimized chat interface"""
    st.markdown("### 💬 Chat Assistant")
    
    # Voice input result
    voice_input = st.session_state.get('mobile_voice_input', '')
    quick_input = st.session_state.get('mobile_quick_input', '')
    
    # Text input with mobile optimization
    user_input = st.text_area(
        "Your Question:",
        value=voice_input or quick_input,
        placeholder="Ask about solar, wind, biogas, costs, maintenance...",
        height=100,
        key="mobile_text_input"
    )
    
    # Clear session state
    if voice_input:
        st.session_state.mobile_voice_input = ''
    if quick_input:
        st.session_state.mobile_quick_input = ''
    
    # Submit button
    if st.button("📤 Send Question", type="primary", key="mobile_submit"):
        if user_input.strip():
            with st.spinner("🔍 Finding the best answer..."):
                result = chatbot.query_rag(user_input)
                
                # Store for voice playback
                st.session_state.last_response = result['answer']
                
                # Display response
                st.markdown("#### 🤖 Answer:")
                st.markdown(f"""
                <div class="bot-response mobile-text">
                    {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Mobile-friendly metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Confidence", f"{result['confidence']:.0%}")
                with col2:
                    st.metric("⏱️ Time", f"{result['processing_time']:.1f}s")
                with col3:
                    st.metric("📚 Sources", len(result['sources']))
                
                if result['sources']:
                    st.caption(f"📖 Based on: {', '.join(result['sources'])}")

def mobile_offline_mode():
    """Handle offline functionality"""
    st.markdown("### 📱 Offline Mode")
    
    # Check if running offline
    try:
        import requests
        requests.get("https://www.google.com", timeout=5)
        online_status = True
    except:
        online_status = False
    
    if online_status:
        st.success("🌐 Online - Full functionality available")
    else:
        st.warning("📴 Offline mode - Using cached data")
        st.info("💡 Tip: Downloaded content is still available for queries")

def main_mobile():
    """Main mobile application"""
    st.set_page_config(
        page_title="🌱 Renewable Energy Bot",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_mobile_css()
    
    # PWA Header
    st.markdown("""
    <div class="pwa-header">
        <h2>🌱 Renewable Energy Assistant</h2>
        <p>Clean Energy Solutions for Rural Communities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    if not chatbot:
        st.error("❌ Unable to initialize. Please check your connection.")
        return
    
    # Offline status
    mobile_offline_mode()
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎤 Voice", "📋 Quick"])
    
    with tab1:
        mobile_chat_interface(chatbot)
    
    with tab2:
        mobile_voice_interface(chatbot)
    
    with tab3:
        quick_action_buttons()
        
        # Show recent questions
        if chatbot.chat_history:
            st.markdown("### 📝 Recent Questions")
            for i, chat in enumerate(reversed(chatbot.chat_history[-3:])):
                with st.expander(f"💭 {chat['query'][:30]}..."):
                    st.write(chat['response'][:200] + "..." if len(chat['response']) > 200 else chat['response'])
    
    # Bottom navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", key="nav_home"):
            st.experimental_rerun()
    with col2:
        if st.button("📊 Stats", key="nav_stats"):
            if chatbot.chat_history:
                st.metric("Total Queries", len(chatbot.chat_history))
    with col3:
        if st.button("ℹ️ About", key="nav_about"):
            st.info("🌱 AI Assistant for Rural Renewable Energy v1.0")

if __name__ == "__main__":
    main_mobile()
