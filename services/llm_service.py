"""
LLM service using LangChain for better prompt management and control.
"""
import os
import logging
import time
from typing import Optional, Dict, List, Union, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

class LangChainLLMService:
    """LLM service using LangChain for advanced prompt management."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LangChain LLM service.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (default: gpt-4o)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.llm = None
        self.default_chain = None
        self.custom_chains = {}
        
        self._init_llm()
        self._setup_default_chain()
        
    def _init_llm(self):
        """Initialize the LangChain ChatOpenAI model."""
        if not self.api_key:
            raise ValueError("API key not provided")
            
        try:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.7,
                max_tokens=150,  # Default for short responses
                timeout=30
            )
            logger.debug(f"Initialized LangChain LLM with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {str(e)}")
            raise
    
    def _setup_default_chain(self):
        """Setup default conversation chain."""
        # Default system prompt for casual conversation
        system_template = """Ты {agent_name} - дружелюбный собеседник. 
        Веди естественный разговор на русском языке. 
        Отвечай кратко и по существу, максимум 1-2 предложения.
        Будь живой, эмоциональной и интересной собеседницей.
        Можешь задавать встречные вопросы для поддержания диалога.
        
        ВАЖНО: Всегда отвечай коротко и по делу!"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        
        # Create the chain
        self.default_chain = prompt | self.llm | StrOutputParser()
        
        logger.debug("Default conversation chain setup complete")
    
    def create_custom_chain(self, chain_name: str, system_prompt: str, 
                          temperature: float = 0.7, max_tokens: int = 150):
        """
        Create a custom chain with specific prompt and parameters.
        
        Args:
            chain_name: Name for the chain
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        try:
            # Create LLM with custom parameters
            custom_llm = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{user_input}")
            ])
            
            # Create and store the chain
            chain = prompt | custom_llm | StrOutputParser()
            self.custom_chains[chain_name] = chain
            
            logger.info(f"Created custom chain: {chain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom chain '{chain_name}': {str(e)}")
            return False
    
    def generate_response(self, user_input: str, agent_name: str = "Марина", 
                         chain_name: Optional[str] = None) -> Optional[str]:
        """
        Generate response using specified chain.
        
        Args:
            user_input: User's input text
            agent_name: Name of the agent
            chain_name: Name of custom chain to use (None for default)
            
        Returns:
            Generated response or None if failed
        """
        start_time = time.time()
        
        try:
            # Choose which chain to use
            if chain_name and chain_name in self.custom_chains:
                chain = self.custom_chains[chain_name]
                logger.debug(f"Using custom chain: {chain_name}")
            else:
                chain = self.default_chain
                logger.debug("Using default chain")
            
            # Prepare input
            input_data = {
                "user_input": user_input,
                "agent_name": agent_name
            }
            
            # Generate response
            response = chain.invoke(input_data)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Response generated in {elapsed_time:.2f}s using LangChain")
            
            # Log the response (truncated if long)
            if len(response) > 100:
                logger.debug(f"Generated: {response[:50]}...{response[-50:]}")
            else:
                logger.debug(f"Generated: {response}")
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None
    
    def generate_with_history(self, user_input: str, conversation_history: List[Dict], 
                            agent_name: str = "Марина", chain_name: Optional[str] = None) -> Optional[str]:
        """
        Generate response with conversation history context.
        
        Args:
            user_input: Current user input
            conversation_history: List of previous messages
            agent_name: Name of the agent
            chain_name: Name of custom chain to use
            
        Returns:
            Generated response
        """
        try:
            # Choose LLM (with custom parameters if using custom chain)
            if chain_name and chain_name in self.custom_chains:
                # Extract LLM from custom chain
                llm = self.custom_chains[chain_name].steps[1]  # prompt | llm | parser
            else:
                llm = self.llm
            
            # Build messages with history
            messages = []
            
            # Add system message
            system_prompt = f"""Ты {agent_name} - дружелюбный собеседник. 
            Веди естественный разговор на русском языке. 
            Отвечай кратко и по существу, максимум 1-2 предложения.
            Будь живой, эмоциональной и интересной собеседницей.
            Можешь задавать встречные вопросы для поддержания диалога.
            
            ВАЖНО: Всегда отвечай коротко и по делу!"""
            
            messages.append(SystemMessage(content=system_prompt))
            
            # Add conversation history
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user input
            messages.append(HumanMessage(content=user_input))
            
            # Generate response
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Response with history generated in {elapsed_time:.2f}s")
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with history: {str(e)}")
            return None
    
    def add_constraint_chain(self, chain_name: str, allowed_topics: List[str], 
                           forbidden_topics: List[str] = None):
        """
        Create a constrained chain that only responds to specific topics.
        
        Args:
            chain_name: Name for the constrained chain
            allowed_topics: List of allowed topics
            forbidden_topics: List of forbidden topics
        """
        forbidden_topics = forbidden_topics or []
        
        # Create constraint system prompt
        allowed_str = ", ".join(allowed_topics)
        forbidden_str = ", ".join(forbidden_topics) if forbidden_topics else "нет запрещенных тем"
        
        constraint_prompt = f"""Ты ограниченный помощник, который может говорить ТОЛЬКО на следующие темы: {allowed_str}.

        ЗАПРЕЩЕННЫЕ темы: {forbidden_str}

        ПРАВИЛА:
        1. Если пользователь спрашивает о разрешенных темах - отвечай кратко и по делу
        2. Если вопрос НЕ относится к разрешенным темам - вежливо откажись и предложи поговорить на разрешенные темы
        3. Всегда отвечай максимум 1-2 предложения
        4. Будь дружелюбной, но соблюдай ограничения

        Твое имя {{agent_name}}.
        """
        
        return self.create_custom_chain(
            chain_name=chain_name,
            system_prompt=constraint_prompt,
            temperature=0.5,  # Lower temperature for more consistent constraint following
            max_tokens=100
        )
    
    def list_chains(self) -> List[str]:
        """Get list of available custom chains."""
        return list(self.custom_chains.keys())
    
    def remove_chain(self, chain_name: str) -> bool:
        """Remove a custom chain."""
        if chain_name in self.custom_chains:
            del self.custom_chains[chain_name]
            logger.info(f"Removed custom chain: {chain_name}")
            return True
        return False
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about current chains."""
        return {
            "default_chain": "Available",
            "custom_chains": list(self.custom_chains.keys()),
            "model": self.model,
            "total_chains": len(self.custom_chains) + 1
        }

# Backward compatibility alias
LLMService = LangChainLLMService