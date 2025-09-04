from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, task
from typing import List
from .tools.rag_tool import rag_ingest, rag_retrieve


@CrewBase
class RagCrew:
    
    agents: List[Agent]
    tasks: List[Task]
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def rag_ingest(self):
        return rag_ingest
    
    def rag_retrieve(self):
        return rag_retrieve
    
    def __init__(self):
        self.load_configurations()  # ensure YAML is loaded
        self.map_all_agent_variables()
        self.map_all_task_variables()
    
    # -----------------------------
    # Agent Definition
    # -----------------------------
    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],
            tools=[rag_ingest, rag_retrieve],  # Add tools here
            verbose=True
        )
    
    # -----------------------------
    # Tasks
    # -----------------------------
    @task
    def ingest_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_ingest_task"],
            agent=self.rag_agent(),
            tools=[rag_ingest]  # Specify the specific tool for this task
        )
    
    @task
    def retrieve_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_retrieve_task"],
            agent=self.rag_agent(),
            tools=[rag_retrieve]  # Specify the specific tool for this task
        )
    
    # -----------------------------
    # Specialized Crew Builders
    # -----------------------------
    def ingest_crew(self) -> Crew:
        """Creates a Crew specifically for the ingestion task."""
        return Crew(
            agents=[self.rag_agent()],
            tasks=[self.ingest_task()],
            verbose=True
        )
        
    def retrieve_crew(self) -> Crew:
        """Creates a Crew specifically for the retrieval task."""
        return Crew(
            agents=[self.rag_agent()],
            tasks=[self.retrieve_task()],
            verbose=True
        )