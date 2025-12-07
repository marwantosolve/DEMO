"""
Research Multi-Agent System using Google ADK.

A complete example of a research MAS with 4 specialized agents:
1. Orchestrator - Coordinates the research task
2. Academic Researcher - Searches academic/scholarly sources
3. Industry Researcher - Searches industry/commercial sources  
4. Writer - Synthesizes findings into a report

Uses gemini-2.0-flash model for all agents.
"""

from typing import List, Dict, Any, Optional
import os


def search_academic_sources(query: str, max_results: int = 3) -> dict:
    """
    Search academic and scholarly sources for research.
    
    Args:
        query: The search query for academic papers
        max_results: Maximum number of results to return
        
    Returns:
        dict: Search results with status and papers found
    """
    # Simulated academic search results for demonstration
    # In production, this would connect to APIs like Semantic Scholar, arXiv, etc.
    sample_results = {
        "ai healthcare": [
            {
                "title": "Deep Learning for Medical Image Analysis",
                "source": "Nature Medicine, 2023",
                "summary": "Comprehensive review of CNN architectures for radiology, pathology, and dermatology. Achieves 94% accuracy in detecting lung nodules."
            },
            {
                "title": "Transformer Models in Clinical Decision Support",
                "source": "JAMA, 2024",
                "summary": "BERT-based models for clinical text analysis show 89% accuracy in predicting patient outcomes."
            },
            {
                "title": "Federated Learning for Privacy-Preserving Healthcare AI",
                "source": "Lancet Digital Health, 2023",
                "summary": "Novel approach enabling multi-hospital AI training without sharing patient data."
            }
        ],
        "default": [
            {
                "title": f"Academic Study on: {query}",
                "source": "Journal of Research, 2024",
                "summary": f"Comprehensive academic research covering various aspects of {query}."
            }
        ]
    }
    
    key = query.lower()
    results = sample_results.get(key, sample_results["default"])[:max_results]
    
    return {
        "status": "success",
        "query": query,
        "num_results": len(results),
        "papers": results
    }


def search_industry_sources(query: str, max_results: int = 3) -> dict:
    """
    Search industry and commercial sources for market insights.
    
    Args:
        query: The search query for industry information
        max_results: Maximum number of results to return
        
    Returns:
        dict: Search results with status and industry insights
    """
    # Simulated industry search results
    sample_results = {
        "ai healthcare": [
            {
                "company": "Google Health",
                "insight": "Partnering with major hospital networks for AI-assisted diagnostics. DeepMind's AlphaFold being applied to drug discovery."
            },
            {
                "company": "Microsoft Healthcare",
                "insight": "Azure Health Bot deployed in 500+ healthcare organizations. DAX Copilot reducing clinical documentation time by 50%."
            },
            {
                "company": "NVIDIA Clara",
                "insight": "AI platform for medical imaging deployed in 1000+ institutions globally. Market valued at $12B by 2025."
            }
        ],
        "default": [
            {
                "company": "Industry Leader",
                "insight": f"Major developments in {query} sector with growing market adoption."
            }
        ]
    }
    
    key = query.lower()
    results = sample_results.get(key, sample_results["default"])[:max_results]
    
    return {
        "status": "success",
        "query": query,
        "num_results": len(results),
        "insights": results
    }


def write_report_section(
    section_title: str,
    content: str,
    style: str = "formal"
) -> dict:
    """
    Write a section of the research report.
    
    Args:
        section_title: Title of the report section
        content: Main content to include in the section
        style: Writing style (formal, casual, technical)
        
    Returns:
        dict: The formatted report section
    """
    return {
        "status": "success",
        "section": {
            "title": section_title,
            "content": content,
            "style": style,
            "word_count": len(content.split())
        }
    }


class ResearchMAS:
    """
    Research Multi-Agent System using Google ADK.
    
    Creates a team of specialized agents that collaborate to produce
    comprehensive research reports on any topic.
    
    Usage:
        mas = ResearchMAS()
        root_agent = mas.create_agents()
        
        # Use with ADK adapter for evaluation
        from mas_eval.adapters import ADKAdapter
        adapter = ADKAdapter(root_agent)
        result, spans = await adapter.run_with_tracing("Research AI in Healthcare")
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None
    ):
        """
        Initialize the Research MAS.
        
        Args:
            model: The Gemini model to use for all agents
            api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._agents = {}
        self._root_agent = None
    
    def create_agents(self):
        """
        Create and configure all research agents.
        
        Returns:
            The root orchestrator agent with sub-agents configured
        """
        try:
            from google.adk.agents import Agent
        except ImportError:
            raise ImportError(
                "google-adk required. Install with: pip install google-adk"
            )
        
        # Academic Researcher Agent
        academic_researcher = Agent(
            name="Academic_Researcher",
            model=self.model,
            description=(
                "Expert at finding and analyzing academic papers, scholarly articles, "
                "and peer-reviewed research. Focuses on scientific rigor and citations."
            ),
            instruction=(
                "You are an academic researcher. Your role is to:\n"
                "1. Search for peer-reviewed papers and scholarly sources\n"
                "2. Analyze research methodology and findings\n"
                "3. Identify key citations and influential works\n"
                "4. Summarize findings with academic rigor\n\n"
                "Always cite your sources and note the publication venue. "
                "Focus on recent publications (last 5 years) unless historical context is needed."
            ),
            tools=[search_academic_sources]
        )
        
        # Industry Researcher Agent
        industry_researcher = Agent(
            name="Industry_Researcher",
            model=self.model,
            description=(
                "Expert at researching industry trends, company developments, "
                "market analysis, and commercial applications."
            ),
            instruction=(
                "You are an industry analyst. Your role is to:\n"
                "1. Research major companies and their initiatives\n"
                "2. Analyze market trends and projections\n"
                "3. Identify commercial applications and products\n"
                "4. Assess competitive landscape\n\n"
                "Focus on actionable business insights and real-world applications. "
                "Include market size data and growth projections when available."
            ),
            tools=[search_industry_sources]
        )
        
        # Writer Agent
        writer = Agent(
            name="Writer",
            model=self.model,
            description=(
                "Expert at synthesizing research from multiple sources into "
                "clear, well-structured reports."
            ),
            instruction=(
                "You are a professional research writer. Your role is to:\n"
                "1. Synthesize findings from academic and industry research\n"
                "2. Structure content into clear sections\n"
                "3. Write in a professional, accessible style\n"
                "4. Highlight key insights and implications\n\n"
                "Create reports that are both informative and engaging. "
                "Balance depth with readability."
            ),
            tools=[write_report_section]
        )
        
        # Orchestrator Agent (root)
        orchestrator = Agent(
            name="Orchestrator",
            model=self.model,
            description=(
                "Coordinates research tasks across specialized agents to produce "
                "comprehensive, high-quality research reports."
            ),
            instruction=(
                "You are the research coordinator. Your responsibilities:\n"
                "1. Understand the research request and break it into subtasks\n"
                "2. Delegate academic research to Academic_Researcher\n"
                "3. Delegate industry research to Industry_Researcher\n"
                "4. Have Writer synthesize all findings into a cohesive report\n"
                "5. Ensure all parts of the request are addressed\n\n"
                "WORKFLOW:\n"
                "- First, think about what aspects need to be researched\n"
                "- Assign specific queries to the appropriate researcher\n"
                "- Once research is complete, coordinate with Writer\n"
                "- Review the final output for completeness\n\n"
                "Always ensure both academic rigor AND practical relevance."
            ),
            sub_agents=[academic_researcher, industry_researcher, writer]
        )
        
        self._agents = {
            "orchestrator": orchestrator,
            "academic_researcher": academic_researcher,
            "industry_researcher": industry_researcher,
            "writer": writer
        }
        self._root_agent = orchestrator
        
        return orchestrator
    
    def get_agent(self, name: str):
        """Get a specific agent by name."""
        return self._agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all configured agents."""
        return self._agents.copy()
    
    @property
    def root_agent(self):
        """Get the root orchestrator agent."""
        if self._root_agent is None:
            self.create_agents()
        return self._root_agent


def create_research_mas(
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None
) -> ResearchMAS:
    """
    Factory function to create a Research MAS.
    
    Args:
        model: The Gemini model to use
        api_key: Optional API key
        
    Returns:
        Configured ResearchMAS instance
    """
    mas = ResearchMAS(model=model, api_key=api_key)
    mas.create_agents()
    return mas
