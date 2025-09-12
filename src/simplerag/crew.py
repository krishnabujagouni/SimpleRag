from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, task
from typing import List
from simplerag.tools.rag_tool import rag_ingest, rag_retrieve
@CrewBase
class RagCrew:
    
    agents: List[Agent]
    tasks: List[Task]
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def __init__(self):
        self.load_configurations()
        self.map_all_agent_variables()
        self.map_all_task_variables()
    
    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],
            tools=[rag_ingest, rag_retrieve],
            verbose=True
        )
    
    @task
    def ingest_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_ingest_task"],
            agent=self.rag_agent(),
            tools=[rag_ingest]
        )
    
    @task
    def retrieve_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_retrieve_task"],
            agent=self.rag_agent(),
            tools=[rag_retrieve]
        )
    
    def ingest_crew(self) -> Crew:
        return Crew(
            agents=[self.rag_agent()],
            tasks=[self.ingest_task()],
            verbose=True
        )
        
    def retrieve_crew(self) -> Crew:
        return Crew(
            agents=[self.rag_agent()],
            tasks=[self.retrieve_task()],
            verbose=True
        )